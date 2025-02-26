"""
Does a direct solve for matrix systems where the right-hand-side and solution vectors can
be decomposed into locally-coupled blocks and joining elements, so that any element in a
'locally-coupled block' is not coupled (by a non-zero matrix entry) to any other
'locally-coupled block' except its own, but may be coupled to any of the 'joining elements'.

Matrices of this type often result from finite element discretizations, where the degrees
of freedom within the volume of an element (or contiguous group of elements) are coupled
to themselves, but only couple to another element via the degrees of freedom on the
surface shared by both elements. The 'locally coupled blocks' are then the interiors of
the elements, and the 'joining elements' are those on the surfaces of elements.

Using an algorithm suggested by the MFEM documentation
(https://docs.mfem.org/html/classmfem_1_1StaticCondensation.html), write the full matrix
system as
```math
\\begin{align}
A\\cdot X &= U
```
By reordering the entries of X and B so that the 'local blocks' are the first entries,
with each local block being a continuous chunk, followed by the 'joining elements', the
matrix system can be rewritten as
```math
\\begin{align}
\\left(\\begin{array}{cc}
a & b\\\\
c & d\\\\
\\end{array}\\right)\\cdot\\left(\\begin{array}{c}
x\\\\
y\\\\
\\end{array}\\right)=\\left(\\begin{array}{c}
u\\\\
v\\\\
\\end{array}\\right)
\\end{align}
```
In this form, \$a\$ is block-diagonal so \$a\\cdot x = u\$ can be solved efficiently, and
parallelised. The remaining part of the solution is found by forming the Schur complement
of \$a\$, doing a matrix-solve using that, and back-substituting, as follows.
```math
\\begin{align}
& a\\cdot x + b \\cdot y = u \\\\
& x = A^{-1}\\cdot u - A^{-1} \\cdot b \\cdot y \\\\
& c\\cdot x + d\\cdot y = v \\\\
& c\\cdot (A^{-1}\\cdot u - A^{-1} \\cdot b \\cdot y) + d\\cdot y = v \\\\
& (d - c\\cdot A^{-1} \\cdot b) \\cdot y = v - c\\cdot A^{-1}\\cdot u \\\\
& s\\cdot y = v - A^{-1}\\cdot u \\\\
\\end{align}
```
where \$s = (d - c\\cdot A^{-1} \\cdot b)\$ is the 'Schur complement' of \$a\$. Once \$y\$
is known, we can substitute back into the expression above for \$x\$
```math
\\begin{align}
& x = A^{-1}\\cdot u - A^{-1} \\cdot b \\cdot y \\\\
\\end{align}
```
"""
module MPIStaticCondensation

export CondensedFactorization, static_condensed_solve, ldiv!,
       update_condensed_factorization!

using LinearAlgebra
import LinearAlgebra: ldiv!
using MPI
using SparseArrays
using TimerOutputs

mutable struct CondensedFactorization{T, M<:AbstractMatrix{T}, F1<:Factorization{T},
                                      F2<:Union{Factorization{T},Nothing},
                                      LB<:Union{Vector{Vector{Int}},Vector{UnitRange{Int}}},
                                      C<:Union{MPI.Comm,Nothing},
                                      SV<:AbstractVector,
                                      V<:Union{AbstractVector,Nothing}} <: Factorization{T}
  n::Int
  n_my_blocks::Int
  sparse_local_blocks::Bool
  localblock_factorizations::Vector{F1}
  local_blocks::LB
  my_blocks::LB
  joining_elements::Vector{Int}
  my_joining_elements::Vector{Int}
  joining_elements_chunk::UnitRange{Int}
  split_c::Vector{M}
  split_ainv_dot_b::Vector{M}
  schur_complement_factorization::F2
  split_rhs_local::Vector{Vector{T}}
  split_solution_local::Vector{Vector{T}}
  joining_elements_rhs::SV
  joining_elements_rhs_unshared_buffer::V
  joining_elements_solution::SV
  shared_comm::C
  shared_comm_size::Int
  shared_comm_rank::Int
end

function CondensedFactorization(A::AbstractMatrix{T}, localblocksize::Integer, couplingblocksize::Integer; kwargs...) where T
  n = size(A, 1)
  ncouplingblocks = (n - localblocksize) ÷ (localblocksize + couplingblocksize)
  nlocalblocks = ncouplingblocks + 1
  return CondensedFactorization(A, fill(localblocksize, nlocalblocks), fill(couplingblocksize, ncouplingblocks); kwargs...)
end

function CondensedFactorization(A::AbstractMatrix{T}, localblocksizes::Vector{<:Integer}, couplingblocksizes::Vector{<:Integer}; kwargs...) where T
    c = 1
    local_blocks = [c:c+localblocksizes[1]-1]
    c += localblocksizes[1]
    for i in 1:length(localblocksizes)-1
        c += couplingblocksizes[i]
        push!(local_blocks, c:c+localblocksizes[i+1]-1)
        c += localblocksizes[i+1]
    end
    return CondensedFactorization(A, local_blocks; kwargs...)
end

function CondensedFactorization(A::AbstractMatrix{T},
                                local_blocks::Union{Vector{Vector{I}},Vector{UnitRange{I}}} where I <: Integer;
                                sparse_local_blocks=false,
                                shared_MPI_comm::Union{MPI.Comm,Nothing}=nothing,
                                joining_elements_rhs_buffer::Union{AbstractVector,Nothing}=nothing,
                                joining_elements_solution_buffer::Union{AbstractVector,Nothing}=nothing) where T
  n = size(A, 1)
  @assert n == size(A, 2)

  n_local_blocks = length(local_blocks)
  all_indices = collect(1:n)
  all_local_block_indices = union((inds isa UnitRange ? collect(inds) : inds for inds in local_blocks)...)
  joining_elements = [i for i in all_indices if !(i in all_local_block_indices)]

  if (joining_elements_rhs_buffer !== nothing
      && length(joining_elements_rhs_buffer) != length(joining_elements))
    error("The length of `joining_elements_rhs_buffer` "
          * "($(length(joining_elements_rhs_buffer))) must be equal to the number "
          * "of joining elements ($(length(joining_elements)))")
  end
  if (joining_elements_solution_buffer !== nothing
      && length(joining_elements_solution_buffer) != length(joining_elements))
    error("The length of `joining_elements_solution_buffer` "
          * "($(length(joining_elements_solution_buffer))) must be equal to the number "
          * "of joining elements ($(length(joining_elements)))")
  end

  if shared_MPI_comm !== nothing
    shared_comm_size = MPI.Comm_size(shared_MPI_comm)
    shared_comm_rank = MPI.Comm_rank(shared_MPI_comm)

    if shared_comm_size > n_local_blocks
      if shared_comm_rank < n_local_blocks
        my_blocks = local_blocks[shared_comm_rank+1:shared_comm_rank+1]
      else
        my_blocks = local_blocks[1:0]
      end
    elseif n_local_blocks % shared_comm_size == 0
      blocks_per_rank = n_local_blocks ÷ shared_comm_size
      my_blocks_inds = shared_comm_rank*blocks_per_rank+1:(shared_comm_rank+1)*blocks_per_rank
      my_blocks = local_blocks[my_blocks_inds]
    else
      error("Number of blocks ($n_local_blocks) must be divisible by the number of MPI "
            * "ranks ($shared_comm_size), unless there are more MPI ranks than blocks.")
    end
    n_my_blocks = length(my_blocks)

    if joining_elements_rhs_buffer === nothing
      error("When using shared-memory MPI, must pass a shared-memory buffer to the "
            * "`joining_elements_rhs_buffer` argument of `CondensedFactorization()`")
    end
    joining_elements_rhs = joining_elements_rhs_buffer

    if joining_elements_solution_buffer === nothing
      error("When using shared-memory MPI, must pass a shared-memory buffer to the "
            * "`joining_elements_solution_buffer` argument of `CondensedFactorization()`")
    end
    joining_elements_solution = joining_elements_solution_buffer

    joining_elements_chunk_size = (length(joining_elements) + shared_comm_size - 1) ÷ shared_comm_size
    joining_elements_chunk = (shared_comm_rank * joining_elements_chunk_size + 1):min((shared_comm_rank + 1) * joining_elements_chunk_size, length(joining_elements))
    my_joining_elements = joining_elements[joining_elements_chunk]
    joining_elements_rhs_unshared_buffer = zeros(eltype(A), length(joining_elements_rhs))
  else
    shared_comm_size = 1
    shared_comm_rank = 0
    n_my_blocks = n_local_blocks
    my_blocks = local_blocks
    my_blocks_inds = 1:n_local_blocks
    if joining_elements_rhs_buffer === nothing
      joining_elements_rhs = similar(A, length(joining_elements))
    else
      joining_elements_rhs = joining_elements_rhs_buffer
    end
    if joining_elements_solution_buffer === nothing
      joining_elements_solution = similar(A, length(joining_elements))
    else
      joining_elements_solution = joining_elements_solution_buffer
    end
    my_joining_elements = joining_elements
    joining_elements_chunk = 1:length(joining_elements)
    joining_elements_rhs_unshared_buffer = nothing
  end

  @inbounds @boundscheck begin
    # When `--check-bounds=yes`, verify that all the matrix elements that are supposed to
    # be zero are actually zero.
    for myblockx ∈ 1:n_my_blocks, iblocky ∈ 1:n_local_blocks
      iblockx = my_blocks_inds[myblockx]
      if iblockx == iblocky
        # This is a local block that should have non-zero elements
        continue
      end
      this_block = A[my_blocks[myblockx], local_blocks[iblocky]]
      if !all(this_block .== 0.0)
        error("In block ($iblockx,$iblocky), with row indices $(my_blocks[iblockx]) and "
              * "column indices $(local_blocks[iblocky]), found non-zero entries where "
              * "there should not be any.")
      end
    end
  end

  # Check indices were unique, so all indices are now in either local_blocks or
  # joining_elements but not both.
  if !all(1 ≤ i ≤ n for inds in local_blocks for i in inds)
    error("local_blocks contains indices outside the range of indices in A")
  end
  if sum(length(inds) for inds in local_blocks) + length(joining_elements) != n
    error("Total number of indices not equal to the size of A")
  end

  all_block_sizes = [length(inds) for inds in my_blocks]
  split_rhs_local = [similar(A, nblock) for nblock ∈ all_block_sizes]
  split_solution_local = [similar(A, nblock) for nblock ∈ all_block_sizes]

  split_b = [A[inds, joining_elements] for inds in my_blocks]
  if sparse_local_blocks
    split_c = [sparse(@view(A[joining_elements, inds])) for inds in my_blocks]
    localblock_factorizations = [lu(sparse(@view(A[inds,inds]))) for inds in my_blocks]
  else
    split_c = [A[joining_elements, inds] for inds in my_blocks]
    localblock_factorizations = [lu(@view(A[inds,inds])) for inds in my_blocks]
  end

  dense_ainv_dot_b = [similar(local_b) for local_b in split_b]
  for (aib, local_ainv, local_b) in zip(dense_ainv_dot_b, localblock_factorizations, split_b)
    for icol in 1:size(local_b, 2)
      this_col = @view local_b[:,icol]
      if any(this_col .!= 0.0)
        @views ldiv!(aib[:,icol], local_ainv, this_col)
      else
        aib[:,icol] .= 0.0
      end
    end
  end
  if sparse_local_blocks
    split_ainv_dot_b = [sparse(local_ainv_dot_b) for local_ainv_dot_b in dense_ainv_dot_b]
  else
    split_ainv_dot_b = dense_ainv_dot_b
  end

  if shared_comm_rank == shared_comm_size - 1
    # Last rank on the communicator does the Schur-complement solve.
    schur_complement = A[joining_elements, joining_elements]
    for (local_c, local_ainv_dot_b) in zip(split_c, dense_ainv_dot_b)
      mul!(schur_complement, local_c, local_ainv_dot_b, -1.0, 1.0)
    end
    if shared_MPI_comm !== nothing
        MPI.Reduce!(schur_complement, +, shared_MPI_comm; root=shared_comm_size-1)
    end
    if size(schur_complement) == (0, 0)
      schur_complement_factorization = nothing
    elseif sparse_local_blocks
      schur_complement_factorization = lu(sparse(schur_complement))
    else
      schur_complement_factorization = lu(schur_complement)
    end
  else
    # Other ranks need to add contributions to Schur-complement matrix, and pass these to
    # last rank.
    schur_complement = zeros(length(joining_elements), length(joining_elements))
    for (local_c, local_ainv_dot_b) in zip(split_c, dense_ainv_dot_b)
      mul!(schur_complement, local_c, local_ainv_dot_b, -1.0, 1.0)
    end
    MPI.Reduce!(schur_complement, +, shared_MPI_comm; root=shared_comm_size-1)
    schur_complement_factorization = nothing
  end

  return CondensedFactorization(n, n_my_blocks, sparse_local_blocks,
                                localblock_factorizations, local_blocks, my_blocks,
                                joining_elements, my_joining_elements,
                                joining_elements_chunk, split_c, split_ainv_dot_b,
                                schur_complement_factorization, split_rhs_local,
                                split_solution_local, joining_elements_rhs,
                                joining_elements_rhs_unshared_buffer,
                                joining_elements_solution, shared_MPI_comm,
                                shared_comm_size, shared_comm_rank)
end
Base.size(A::CondensedFactorization) = (A.n, A.n)
Base.size(A::CondensedFactorization, i) = A.n

@inline function _check_array_zeros(cf, A, timer)
  @boundscheck begin
@timeit timer "check array zeros" begin
    # When `--check-bounds=yes`, verify that all the matrix elements that are supposed to
    # be zero are actually zero.
    local_blocks = cf.local_blocks
    my_blocks = cf.my_blocks
    blocks_per_rank = length(local_blocks) ÷ cf.shared_comm_size
    my_blocks_inds = cf.shared_comm_rank*blocks_per_rank+1:(cf.shared_comm_rank+1)*blocks_per_rank
    for myblockx ∈ 1:length(my_blocks), iblocky ∈ 1:length(local_blocks)
      iblockx = my_blocks_inds[myblockx]
      if iblockx == iblocky
        # This is a local block that should have non-zero elements
        continue
      end
      this_block = A[my_blocks[myblockx], local_blocks[iblocky]]
      if !all(this_block .== 0.0)
        error("In block ($iblockx,$iblocky), with row indices $(my_blocks[iblockx]) and "
              * "column indices $(local_blocks[iblocky]), found non-zero entries where "
              * "there should not be any.")
      end
    end
end
  end
  return nothing
end

function update_condensed_factorization!(cf::CondensedFactorization{T}, A::AbstractMatrix{T}, timer=TimerOutput()) where T
  n = cf.n
  @assert size(A) == (n, n)

  my_blocks = cf.my_blocks
  joining_elements = cf.joining_elements
  sparse_local_blocks = cf.sparse_local_blocks

  @inbounds _check_array_zeros(cf, A, timer)

@timeit timer "split b" begin
  split_b = [A[inds, joining_elements] for inds in my_blocks]
end
  if sparse_local_blocks
@timeit timer "sparse localblock factorizations" begin
    cf.split_c .= [sparse(@view(A[joining_elements, inds])) for inds in my_blocks]
    for (i, (fac,inds)) in enumerate(zip(cf.localblock_factorizations, my_blocks))
      sparse_block = sparse(@view(A[inds,inds]))
      try
        lu!(fac, sparse_block; check=false)
      catch e
        if !isa(e, ArgumentError)
          rethrow(e)
        end
        cf.localblock_factorizations[i] = lu(sparse_block)
      end
    end
end
  else
@timeit timer "dense localblock factorizations" begin
    for (c,inds) in zip(cf.split_c, my_blocks)
      c .= @view A[joining_elements, inds]
    end
    for (i, (fac,inds)) in enumerate(zip(cf.localblock_factorizations, my_blocks))
      # Reuse matrix already allocated in fac
      mat = fac.factors
      mat .= @view A[inds,inds]
      cf.localblock_factorizations[i] = lu!(mat)
    end
end
  end

  if sparse_local_blocks
@timeit timer "allocate dense new_ainv_dot_b inv" begin
    new_ainv_dot_b = [similar(local_b) for local_b in split_b]
end
  else
    new_ainv_dot_b = cf.split_ainv_dot_b
  end
@timeit timer "sparse ainv_dot_b inv" begin
  for (aib, local_ainv, local_b) in zip(new_ainv_dot_b, cf.localblock_factorizations, split_b)
    for icol in 1:size(local_b, 2)
      this_col = @view local_b[:,icol]
      if any(this_col .!= 0.0)
        @views ldiv!(aib[:,icol], local_ainv, this_col)
      else
        aib[:,icol] .= 0.0
      end
    end
  end
end
  if sparse_local_blocks
@timeit timer "sparse ainv_dot_b sparsify" begin
    cf.split_ainv_dot_b .= [sparse(local_ainv_dot_b) for local_ainv_dot_b in new_ainv_dot_b]
end
  end

  if length(cf.joining_elements) == 0
    # Nothing to do as Schur complement is empty
  elseif cf.shared_comm_rank == cf.shared_comm_size - 1
@timeit timer "schur_complement mul!" begin
    # Last rank on the communicator does the Schur-complement solve.
    schur_complement = A[joining_elements, joining_elements]
    for (local_c, local_ainv_dot_b) in zip(cf.split_c, new_ainv_dot_b)
      mul!(schur_complement, local_c, local_ainv_dot_b, -1.0, 1.0)
    end
end
@timeit timer "schur_complement Reduce!" begin
    if cf.shared_comm !== nothing
      MPI.Reduce!(schur_complement, +, cf.shared_comm; root=cf.shared_comm_size-1)
    end
end
    if sparse_local_blocks
@timeit timer "schur_complement lu!" begin
      sparse_schur_complement = sparse(schur_complement)
      try
        lu!(cf.schur_complement_factorization, sparse_schur_complement)
      catch e
        if !isa(e, ArgumentError)
          rethrow(e)
        end
        cf.schur_complement_factorization = lu(sparse_schur_complement)
      end
end
    else
@timeit timer "schur_complement lu" begin
      new_factorization = lu(schur_complement)
      cf.schur_complement_factorization.factors .= new_factorization.factors
      cf.schur_complement_factorization.ipiv .= new_factorization.ipiv
      if cf.schur_complement_factorization.info != new_factorization.info
        error("New schur_complement_factorization has info=$(new_factorization.info). "
              * "Expected same as original $(cf.schur_complement_factorization.info)")
      end
end
    end
  else
    # Other ranks need to add contributions to Schur-complement matrix, and pass these to
    # last rank.
@timeit timer "schur_complement mul!" begin
    schur_complement = zeros(length(joining_elements), length(joining_elements))
    for (local_c, local_ainv_dot_b) in zip(cf.split_c, new_ainv_dot_b)
      mul!(schur_complement, local_c, local_ainv_dot_b, -1.0, 1.0)
    end
end
@timeit timer "schur_complement Reduce!" begin
    MPI.Reduce!(schur_complement, +, cf.shared_comm; root=cf.shared_comm_size-1)
end
    schur_complement_factorization = nothing
  end

  return cf
end

function split_rhs!(A::CondensedFactorization, U::AbstractVector)
  n_my_blocks = A.n_my_blocks
  my_blocks = A.my_blocks
  my_joining_elements = A.my_joining_elements
  split_rhs_local = A.split_rhs_local
  joining_elements_rhs = A.joining_elements_rhs
  joining_elements_chunk = A.joining_elements_chunk

  for iblock in 1:n_my_blocks
    split_rhs_local[iblock] .= @view U[my_blocks[iblock]]
  end

  joining_elements_rhs[joining_elements_chunk] .= @view U[my_joining_elements]

  return nothing
end

function localblocks_solve!(A)
  split_rhs_local = A.split_rhs_local
  split_solution_local = A.split_solution_local

  n_my_blocks = A.n_my_blocks
  localblock_factorizations = A.localblock_factorizations

  for iblock in 1:n_my_blocks
    ldiv!(split_solution_local[iblock], localblock_factorizations[iblock],
          split_rhs_local[iblock])
  end

  return nothing
end

function schur_complement_solve!(A, timer)
  joining_elements_solution = A.joining_elements_solution
  if length(joining_elements_solution) == 0
      # No joining elements, so nothing to do.
      return nothing
  end
  n_my_blocks = A.n_my_blocks
  split_solution_local = A.split_solution_local
  split_c = A.split_c
  joining_elements_rhs = A.joining_elements_rhs
  schur_complement_factorization = A.schur_complement_factorization
  shared_comm = A.shared_comm
  shared_comm_rank = A.shared_comm_rank
  shared_comm_size = A.shared_comm_size

  # Add MPI.Barrier() calls to ensure joining_elements_rhs is modified sequentially by
  # each rank.
  if shared_comm === nothing
@timeit timer "split_c mul! serial" begin
    for iblock in 1:n_my_blocks
      mul!(joining_elements_rhs, split_c[iblock], split_solution_local[iblock], -1.0, 1.0)
    end
end
  else
    joining_elements_rhs_unshared_buffer = A.joining_elements_rhs_unshared_buffer
    if shared_comm_rank == 0
@timeit timer "split_c mul! rank0" begin
      for iblock in 1:n_my_blocks
        mul!(joining_elements_rhs, split_c[iblock],
             split_solution_local[iblock], -1.0, 1.0)
      end
end
    else
@timeit timer "split_c mul! rankn" begin
      if n_my_blocks > 0
        mul!(joining_elements_rhs_unshared_buffer, split_c[1],
             split_solution_local[1])
      end
      for iblock in 2:n_my_blocks
        mul!(joining_elements_rhs_unshared_buffer, split_c[iblock],
             split_solution_local[iblock], 1.0, 1.0)
      end
end
@timeit timer "split_c Barriers before rankn" begin
      for r ∈ 1:shared_comm_rank
        MPI.Barrier(shared_comm)
      end
      joining_elements_rhs .-= joining_elements_rhs_unshared_buffer
end
    end
@timeit timer "split_c final Barriers" begin
    for r ∈ shared_comm_rank+1:shared_comm_size - 1
      MPI.Barrier(shared_comm)
    end
end
  end

  # Last rank was the last one to modify joining_element_rhs, so if it does the ldiv!()
  # then we do not need an extra MPI.Barrier first
  if shared_comm_rank == shared_comm_size - 1
    @timeit timer "joining elements ldiv!" ldiv!(joining_elements_solution, schur_complement_factorization, joining_elements_rhs)
  end

  if shared_comm !== nothing
    # Need to ensure ldiv!() is finished before we start x_backsubstitution!()
    MPI.Barrier(shared_comm)
  end

  return nothing
end

function x_backsubstitution!(A)
  n_my_blocks = A.n_my_blocks
  split_solution_local = A.split_solution_local
  joining_elements_solution = A.joining_elements_solution
  split_ainv_dot_b = A.split_ainv_dot_b

  for iblock in 1:n_my_blocks
    mul!(split_solution_local[iblock], split_ainv_dot_b[iblock],
         joining_elements_solution, -1.0, 1.0)
  end

  return nothing
end

function gather_split_solution!(X, A)
  n_my_blocks = A.n_my_blocks
  my_blocks = A.my_blocks
  split_solution_local = A.split_solution_local
  my_joining_elements = A.my_joining_elements
  joining_elements_solution = A.joining_elements_solution
  joining_elements_chunk = A.joining_elements_chunk

  for iblock in 1:n_my_blocks
    X[my_blocks[iblock]] .= split_solution_local[iblock]
  end

  X[my_joining_elements] .= @view joining_elements_solution[joining_elements_chunk]

  return nothing
end

function ldiv!(X::AbstractVector, A::CondensedFactorization, U::AbstractVector, timer=TimerOutput())
  @timeit timer "split_rhs!" split_rhs!(A, U)

  # Compute a^{-1}.u
  @timeit timer "localblocks_solve!" localblocks_solve!(A)

  if A.shared_comm !== nothing
    MPI.Barrier(A.shared_comm)
  end

  # Compute y
  @timeit timer "schur_complement_solve!" schur_complement_solve!(A, timer)

  # Back substitute for final solution for x
  @timeit timer "x_backsubstitution!" x_backsubstitution!(A)

  @timeit timer "gather_split_solution!!" gather_split_solution!(X, A)

  return X
end

function ldiv!(A::CondensedFactorization, U::Union{AbstractVector,AbstractMatrix})
  # It is safe to pass the same array for both RHS and solution
  return ldiv!(U, A, U)
end

function ldiv!(x::AbstractMatrix, A::CondensedFactorization, b::AbstractMatrix)
  @boundscheck size(x) == size(b) || error(BoundsError, " x $(size(x)) and b $(size(b)) are not the same size")
  for icol ∈ 1:size(b,2)
    @views ldiv!(x[:,icol], A, b[:,icol])
  end
  return x
end

end # module MPIStaticCondensation
