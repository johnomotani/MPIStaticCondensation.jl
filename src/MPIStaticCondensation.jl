"""
Does a direct solve for matrices that can be decomposed in the following form, where the
sub-matrices are labelled by `M'iddle, `T'op, `B'ottom, `L'eft, `R'ight, `J'oin, `C'orner,
`O'ther corner
```math
\\begin{align}
\\left(\\begin{array}{ccccccc}
M_{1} & R_{1} & 0 & 0 & 0 & 0\\\\
B_{1} & J_{1} & T_{1} & C_{1} & 0 & 0\\\\
0 & L_{1} & M_{2} & R_{2} & 0 & 0\\\\
0 & O_{1} & B_{2} & J_{2} & T_{2} & C_{2}\\\\
0 & 0 & 0 & L_{2} & M_{3} & R_{3}\\\\
0 & 0 & 0 & O_{2} & B_{3} & J_{3}\\\\
 &  &  &  &  &  & \\ddots
\\end{array}\\right)\\cdot\\left(\\begin{array}{c}
V_{1}\\\\
S_{1}\\\\
V_{2}\\\\
S_{2}\\\\
V_{3}\\\\
S_{3}\\\\
\\vdots
\\end{array}\\right)=\\left(\\begin{array}{c}
\\alpha_{1}\\\\
\\beta_{1}\\\\
\\alpha_{2}\\\\
\\beta_{2}\\\\
\\alpha_{3}\\\\
\\beta_{3}\\\\
\\vdots
\\end{array}\\right)
\\end{align}
```
Matrices like this can be transformed into a reduced matrix solve for just the \$S_{i}\$,
combined with decoupled solves for \$V_{n}\$ that can be done in parallel. This is most
likely to work well when the \$V_{i}\$ are much larger than the \$S_{i}\$, so that the
reduced matrix is much smaller than the original matrix.
"""
module MPIStaticCondensation

export CondensedFactorization, static_condensed_solve, ldiv!

using LinearAlgebra
import LinearAlgebra: ldiv!
using SparseArrays

struct CondensedFactorization{T, M<:AbstractMatrix{T}} <: Factorization{T}
  A::M
  indices::Vector{UnitRange{Int}}
  nlocalblocks::Int
  ncouplingblocks::Int
  reducedlocalindices::Vector{UnitRange{Int}}
  reducedcoupledindices::Vector{UnitRange{Int}}
  localblocksizes::Vector{Int}
  couplingblocksizes::Vector{Int}
end

function CondensedFactorization(A::AbstractMatrix{T}, localblocksize::Integer, couplingblocksize::Integer) where T
  n = size(A, 1)
  ncouplingblocks = (n - localblocksize) ÷ (localblocksize + couplingblocksize)
  nlocalblocks = ncouplingblocks + 1
  return CondensedFactorization(A, fill(localblocksize, nlocalblocks), fill(couplingblocksize, ncouplingblocks))
end

function CondensedFactorization(A::AbstractMatrix{T}, localblocksizes::Vector{<:Integer}, couplingblocksizes::Vector{<:Integer}) where T
  n = size(A, 1)
  ncouplingblocks = length(couplingblocksizes)
  nlocalblocks = length(localblocksizes)

  indices = Vector{UnitRange{Int}}()
  a = 1
  for i = 1:ncouplingblocks
    inds = a:a + localblocksizes[i] - 1
    push!(indices, inds)
    a = a + localblocksizes[i]
    inds = a:a + couplingblocksizes[i] - 1
    push!(indices, inds)
    a = a + couplingblocksizes[i]
  end
  push!(indices, a:a + localblocksizes[end] - 1)
  @assert indices[end][end] == size(A, 1) == size(A, 2)

  reducedlocalindices = Vector{UnitRange{Int}}()
  push!(reducedlocalindices, indices[1])
  for (c, i) in enumerate(3:2:length(indices))
    inds = (indices[i] .- indices[i][1] .+ 1) .+ reducedlocalindices[c][end]
    push!(reducedlocalindices, inds) 
  end
  reducedcoupledindices = Vector{UnitRange{Int}}()
  push!(reducedcoupledindices, indices[2] .- indices[2][1] .+ 1)
  for (c, i) in enumerate(4:2:length(indices))
    inds = (indices[i] .- indices[i][1] .+ 1) .+ reducedcoupledindices[c][end]
    push!(reducedcoupledindices, inds) 
  end
  return CondensedFactorization(A, indices, nlocalblocks, ncouplingblocks, reducedlocalindices, reducedcoupledindices, localblocksizes, couplingblocksizes)
end
Base.size(A::CondensedFactorization) = (size(A.A, 1), size(A.A, 2))
Base.size(A::CondensedFactorization, i) = size(A.A, i)
islocalblock(i) = isodd(i)
iscouplingblock(i) = !islocalblock(i)
localindices(A::CondensedFactorization) = A.indices[1:2:end]
couplingindices(A::CondensedFactorization) = A.indices[2:2:end-1]

function factoriselocals(A::CondensedFactorization{T}) where T
  lis = A.indices
  localfact = lu(A.A[lis[1], lis[1]])
  d = Dict{Int, typeof(localfact)}(1=>localfact)
  for (i, li) in enumerate(lis) # parallelisable
    iscouplingblock(i) && continue
    d[i] = lu(A.A[li, li])
  end
  return d
end

function calculatecouplings(A::CondensedFactorization{T,M}, localfactors) where {T,M}
  d = Dict{Tuple{Int, Int}, M}()
  for (i, li) in enumerate(A.indices) # parallelisable
    islocalblock(i) && continue
    if i - 1 >= 1
      lim = A.indices[i-1]
      d[(i-1, i)] = localfactors[i-1] \ A.A[lim, li]
    end
    if i + 1 <= length(A.indices)
      lip = A.indices[i+1]
      d[(i+1, i)] = localfactors[i+1] \ A.A[lip, li]
    end
  end
  return d
end

function solvelocalparts(A::CondensedFactorization{T,M}, b, localfactors) where {T,M}
  d = Dict{Int, M}()
  for (i, li) in enumerate(A.indices) # parallelisable
    iscouplingblock(i) && continue
    d[i] = localfactors[i] \ b[li, :]
  end
  return d
end

function totallockblocksize(A::CondensedFactorization)
  return sum(length(i) for i in A.reducedlocalindices)
end

function totalcouplingblocksize(A::CondensedFactorization)
  return sum(length(i) for i in A.reducedcoupledindices)
end

function couplingblockindices(A::CondensedFactorization, i)
  @boundscheck @assert i > 1
  return A.indices[i] .- A.indices[i-1][1] .+ 1
end

function localblockindices(A::CondensedFactorization, i)
  @boundscheck @assert i > 1
  return A.indices[i] .- A.indices[i-1][1] .+ 1
end

function assemblecoupledrhs(A::CondensedFactorization, B, localsolutions, couplings)
  b = similar(A.A, totalcouplingblocksize(A))
  c = 0
  for (i, li) in enumerate(A.indices) # parallelisable
    islocalblock(i) && continue
    c += 1
    rows = A.reducedcoupledindices[c]
    b[rows, :] .= B[li, :]
    b[rows, :] .-= A.A[li, A.indices[i-1]] * localsolutions[i-1]
    b[rows, :] .-= A.A[li, A.indices[i+1]] * localsolutions[i+1]
  end
  return b
end

function assemblecoupledlhs(A::CondensedFactorization, couplings)
  M = similar(A.A, totalcouplingblocksize(A), totalcouplingblocksize(A))
  fill!(M, 0)
  c = 0
  for (i, li) in enumerate(A.indices) # parallelisable
    islocalblock(i) && continue
    c += 1
    rows = A.reducedcoupledindices[c]
    M[rows, rows] = A.A[li, li]
    aim = A.A[li, A.indices[i-1]]
    aip = A.A[li, A.indices[i+1]]
    M[rows, rows] .-= aim * couplings[(i - 1, i)]
    M[rows, rows] .-= aip * couplings[(i + 1, i)]
    if c + 1 <= A.ncouplingblocks
      right = A.reducedcoupledindices[c + 1]
      M[rows, right] = A.A[li, A.indices[i + 2]]
      M[rows, right] .-= aip * couplings[(i + 1, i + 2)]
    end
    if c - 1 >= 1
      left = A.reducedcoupledindices[c - 1]
      M[rows, left] = A.A[li, A.indices[i - 2]]
      M[rows, left] .-= aim * couplings[(i - 1, i - 2)]
    end
  end
  return M
end

function coupledx(A::CondensedFactorization{T}, b, localsolutions, couplings) where {T}
  Ac = assemblecoupledlhs(A, couplings)
  bc = assemblecoupledrhs(A, b, localsolutions, couplings)
  xc = Ac \ bc
  x = zeros(T, size(b)...)
  c = 0
  for (i, ind) in enumerate(A.indices)
    islocalblock(i) && continue
    c += 1
    x[ind, :] .= xc[A.reducedcoupledindices[c], :]
  end
  return x
end

function localx(A::CondensedFactorization{T}, xc, b, localsolutions, couplings) where T
  xl = zeros(T, size(b))
  c = 0
  for (i, li) in enumerate(A.indices) # parallelisable
    iscouplingblock(i) && continue
    c += 1
    rows = A.reducedlocalindices[c]
    xl[li, :] .= localsolutions[i]
    for j in (i + 1, i - 1)
      0 < j <= length(A.indices) || continue
      xl[li, :] .-= couplings[(i, j)] * xc[A.indices[j], :]
    end
  end
  return xl
end

function static_condensed_solve(A::CondensedFactorization, b)
  x = similar(b)
  return ldiv!(x, A, b)
end

function ldiv!(x::AbstractVector, A::CondensedFactorization, b::AbstractVector)
  localfactors = factoriselocals(A)
  localsolutions = solvelocalparts(A, b, localfactors)
  couplings = calculatecouplings(A, localfactors)
  xc = coupledx(A, b, localsolutions, couplings)
  xl = localx(A, xc, b, localsolutions, couplings)
  @. x = xl + xc
  return x
end

function ldiv!(x::AbstractMatrix, A::CondensedFactorization, b::AbstractMatrix)
  @boundscheck size(x) == size(b) || error(BoundsError, " x $(size(x)) and b $(size(b)) are not the same size")
  for icol ∈ 1:size(b,2)
    @views ldiv!(x[:,icol], A, b[:,icol])
  end
  return x
end

end # module MPIStaticCondensation
