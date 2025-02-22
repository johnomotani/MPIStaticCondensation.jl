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

export CondensedFactorization, static_condensed_solve, ldiv!

using LinearAlgebra
import LinearAlgebra: ldiv!
using SparseArrays

struct CondensedFactorization{T, M<:AbstractMatrix{T}, F,
                              LB<:Union{Vector{Vector{Int}},Vector{UnitRange{Int}}}} <: Factorization{T}
  n::Int
  n_local_blocks::Int
  localblock_factorizations::Vector{F}
  local_blocks::LB
  joining_elements::Vector{Int}
  split_c::Vector{M}
  split_ainv_dot_b::Vector{M}
  schur_complement_factorization::F
  split_rhs::Vector{Vector{T}}
  split_solution::Vector{Vector{T}}
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
                                sparse_local_blocks=false) where T
  n = size(A, 1)
  @assert n == size(A, 2)
  n_local_blocks = length(local_blocks)

  @inbounds @boundscheck begin
    # When `--check-bounds=yes`, verify that all the matrix elements that are supposed to
    # be zero are actually zero.
    for iblockx ∈ 1:n_local_blocks, iblocky ∈ 1:n_local_blocks
      if iblockx == iblocky
        # This is a local block that should have non-zero elements
        continue
      end
      this_block = A[local_blocks[iblockx], local_blocks[iblocky]]
      if !all(this_block .== 0.0)
        error("In block ($iblockx,$iblocky), with row indices $(local_blocks[iblockx]) and "
              * "column indices $(local_blocks[iblocky]), found non-zero entries where "
              * "there should not be any.")
      end
    end
  end

  all_indices = collect(1:n)
  all_local_block_indices = union((inds isa UnitRange ? collect(inds) : inds for inds in local_blocks)...)
  joining_elements = [i for i in all_indices if !(i in all_local_block_indices)]

  # Check indices were unique, so all indices are now in either local_blocks or
  # joining_elements but not both.
  if !all(1 ≤ i ≤ n for inds in local_blocks for i in inds)
    error("local_blocks contains indices outside the range of indices in A")
  end
  if sum(length(inds) for inds in local_blocks) + length(joining_elements) != n
    error("Total number of indices not equal to the size of A")
  end

  all_block_sizes = [length(inds) for inds in local_blocks]
  push!(all_block_sizes, length(joining_elements))
  split_rhs = [similar(A, nblock) for nblock ∈ all_block_sizes]
  split_solution = [similar(A, nblock) for nblock ∈ all_block_sizes]

  split_b = [A[inds, joining_elements] for inds in local_blocks]
  if sparse_local_blocks
    split_c = [sparse(A[joining_elements, inds]) for inds in local_blocks]
    localblock_factorizations = [lu(sparse(@view(A[inds,inds]))) for inds in local_blocks]
  else
    split_c = [A[joining_elements, inds] for inds in local_blocks]
    localblock_factorizations = [lu(@view(A[inds,inds])) for inds in local_blocks]
  end

  split_ainv_dot_b = [local_aniv \ local_b for (local_aniv, local_b)
                      in zip(localblock_factorizations, split_b)]

  schur_complement = A[joining_elements, joining_elements]
  for (local_c, local_ainv_dot_b) in zip(split_c, split_ainv_dot_b)
    schur_complement .-= local_c * local_ainv_dot_b
  end
  if sparse_local_blocks
    split_ainv_dot_b = [sparse(local_ainv_dot_b) for local_ainv_dot_b in split_ainv_dot_b]
    schur_complement_factorization = lu(sparse(schur_complement))
  else
    schur_complement_factorization = lu(schur_complement)
  end

  return CondensedFactorization(n, n_local_blocks, localblock_factorizations, local_blocks,
                                joining_elements, split_c, split_ainv_dot_b,
                                schur_complement_factorization, split_rhs, split_solution)
end
Base.size(A::CondensedFactorization) = (A.n, A.n)
Base.size(A::CondensedFactorization, i) = A.n

function split_rhs!(A::CondensedFactorization, U::AbstractVector)
  n_local_blocks = A.n_local_blocks
  local_blocks = A.local_blocks
  joining_elements = A.joining_elements
  split_rhs = A.split_rhs

  for iblock in 1:n_local_blocks
    split_rhs[iblock] .= @view U[local_blocks[iblock]]
  end

  split_rhs[end] .= @view U[joining_elements]

  return nothing
end

function localblocks_solve!(A)
  split_rhs = A.split_rhs
  split_solution = A.split_solution

  n_local_blocks = A.n_local_blocks
  localblock_factorizations = A.localblock_factorizations

  for iblock in 1:n_local_blocks
    ldiv!(split_solution[iblock], localblock_factorizations[iblock], split_rhs[iblock])
  end

  return nothing
end

function schur_complement_solve!(A)
  n_local_blocks = A.n_local_blocks
  split_solution = A.split_solution
  split_c = A.split_c
  joining_elements_solution = split_solution[end]
  joining_elements_rhs = A.split_rhs[end]
  schur_complement_factorization = A.schur_complement_factorization

  for iblock in 1:n_local_blocks
    joining_elements_rhs .-= split_c[iblock] * split_solution[iblock]
  end

  ldiv!(joining_elements_solution, schur_complement_factorization, joining_elements_rhs)

  return nothing
end

function x_backsubstitution!(A)
  n_local_blocks = A.n_local_blocks
  split_solution = A.split_solution
  joining_elements_solution = split_solution[end]
  split_ainv_dot_b = A.split_ainv_dot_b

  for iblock in 1:n_local_blocks
    split_solution[iblock] .-= split_ainv_dot_b[iblock] * joining_elements_solution
  end

  return nothing
end

function gather_split_solution!(X, A)
  n_local_blocks = A.n_local_blocks
  local_blocks = A.local_blocks
  split_solution = A.split_solution
  joining_elements = A.joining_elements

  for iblock in 1:n_local_blocks
    X[local_blocks[iblock]] .= split_solution[iblock]
  end

  X[joining_elements] .= split_solution[end]

  return nothing
end

function ldiv!(X::AbstractVector, A::CondensedFactorization, U::AbstractVector)
  split_rhs!(A, U)

  # Compute a^{-1}.u
  localblocks_solve!(A)

  # Compute y
  schur_complement_solve!(A)

  # Back substitute for final solution for x
  x_backsubstitution!(A)

  gather_split_solution!(X, A)

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
