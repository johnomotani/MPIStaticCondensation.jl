using StableRNGs
using Test

using MPIStaticCondensation

@testset "CondensedFactorization" begin
  @testset "sparse=$sparse" for sparse in (false, true)
    @testset "($L,$C)" for (L, C, tol) in (
                                           (2, 1, 1.0e-14),
                                           (3, 2, 4.0e-14),
                                           (4, 2, 1.0e-13),
                                           (16, 4, 1.0e-12),
                                          )

      rng = StableRNG(0)

      A11, A33, A55, A77 = (rand(rng, L, L) for _ in 1:4)
      t12, t32, t34, t54, t56, t76 = (rand(rng, L, C) for _ in 1:6)
      s21, s23, s43, s45, s65, s67 = (rand(rng, C, L) for _ in 1:6)
      a22, a44, a66, a24, a42, a46, a64 = (rand(rng, C, C) for _ in 1:7)
      zLL = zeros(L, L)
      zCL = zeros(C, L)
      zLC = zeros(L, C)
      zCC = zeros(C, C)

      A = [A11 t12 zLL;
           s21 a22 s23;
           zLL t32 A33]
      b = rand(rng, size(A, 1))
      check = A \ b
      Acf = CondensedFactorization(A, L, C; sparse_local_blocks=sparse)
      x = ldiv!(Acf, copy(b))
      @test isapprox(A * x, b; atol=tol)
      @test isapprox(A * check, b; atol=tol)
      @test isapprox(x, check; rtol=tol)

      A = [A11 t12 zLL zLC zLL;
           s21 a22 s23 a24 zCL;
           zLL t32 A33 t34 zLL;
           zCL a42 s43 a44 s45;
           zLL zLC zLL t54 A55]
      b = rand(rng, size(A, 1))
      check = A \ b
      Acf = CondensedFactorization(A, L, C; sparse_local_blocks=sparse)
      x = ldiv!(Acf, copy(b))
      @test isapprox(A * x, b; atol=tol)
      @test isapprox(A * check, b; atol=tol)
      @test isapprox(x, check; rtol=tol)

      A = [A11 t12 zLL zLC zLL zLC zLL;
           s21 a22 s23 a24 zCL zCC zCL;
           zLL t32 A33 t34 zLL zLC zLL;
           zCL a42 s43 a44 s45 a46 zCL;
           zLL zLC zLL t54 A55 t56 zLL;
           zCL zCC zCL a64 s65 a66 s67;
           zLL zLC zLL zLC zLL t76 A77]

      b = rand(rng, size(A, 1))
      check = A \ b
      Acf = CondensedFactorization(A, L, C; sparse_local_blocks=sparse)
      x = ldiv!(Acf, copy(b))
      @test isapprox(A * x, b; atol=tol)
      @test isapprox(A * check, b; atol=tol)
      @test isapprox(x, check; rtol=tol)
    end
  end

  @testset "sparse=$sparse" for sparse in (false, true)
    @testset "($L,$C)" for (L, C, tol) in (
                                           ([3,2], [1], 1.0e-14),
                                           ([3,2,4], [2,3], 1.0e-14),
                                          )

      rng = StableRNG(0)

      A11 = rand(rng, L[1], L[1])
      A33 = rand(rng, L[2], L[2])
      if length(L) > 2
        A55 = rand(rng, L[3], L[3])
      end
      t12 = rand(rng, L[1], C[1])
      t32 = rand(rng, L[2], C[1])
      if length(L) > 2
        t34 = rand(rng, L[2], C[2])
        t54 = rand(rng, L[3], C[2])
      end
      s21 = rand(rng, C[1], L[1])
      s23 = rand(rng, C[1], L[2])
      if length(L) > 2
        s43 = rand(rng, C[2], L[2])
        s45 = rand(rng, C[2], L[3])
      end
      a22 = rand(rng, C[1], C[1])
      if length(L) > 2
        a24 = rand(rng, C[1], C[2])
        a42 = rand(rng, C[2], C[1])
        a44 = rand(rng, C[2], C[2])
      end
      z13 = zeros(L[1], L[2])
      z31 = zeros(L[2], L[1])
      if length(L) > 2
        z15 = zeros(L[1], L[3])
        z35 = zeros(L[2], L[3])
        z51 = zeros(L[3], L[1])
        z53 = zeros(L[3], L[2])
        z25 = zeros(C[1], L[3])
        z41 = zeros(C[2], L[1])
        z14 = zeros(L[1], C[2])
        z52 = zeros(L[3], C[1])
      end

      A = [A11 t12 z13;
           s21 a22 s23;
           z31 t32 A33]
      b = rand(rng, size(A, 1))
      check = A \ b
      Acf = CondensedFactorization(A, L[1:2], C[1:1]; sparse_local_blocks=sparse)
      x = ldiv!(Acf, copy(b))
      @test isapprox(A * x, b; atol=tol)
      @test isapprox(A * check, b; atol=tol)
      @test isapprox(x, check; rtol=tol)

      if length(L) > 2
        A = [A11 t12 z13 z14 z15;
             s21 a22 s23 a24 z25;
             z31 t32 A33 t34 z35;
             z41 a42 s43 a44 s45;
             z51 z52 z53 t54 A55]
        b = rand(rng, size(A, 1))
        check = A \ b
        Acf = CondensedFactorization(A, L[1:3], C[1:2]; sparse_local_blocks=sparse)
        x = ldiv!(Acf, copy(b))
        @test isapprox(A * x, b; atol=tol)
        @test isapprox(A * check, b; atol=tol)
        @test isapprox(x, check; rtol=tol)
      end
    end
  end

  fe_tol = 1.0e-14
  @testset "finite-element like ($nelement_x, $ngrid_x, $nelement_y, $ngrid_y) sparse=$sparse" for
      sparse in (false, true), nelement_x in 1:6, ngrid_x in 3:6, nelement_y in 1:6, ngrid_y in 3:6

    rng = StableRNG(0)

    nx = nelement_x * (ngrid_x - 1) + 1
    ny = nelement_y * (ngrid_y - 1) + 1

    A = zeros(nx * ny, nx * ny)

    # Add an 'x-derivative'
    for irowx in 1:nx, irowy in 1:ny
      irow = (irowx - 1) * ny + irowy
      xe = min((irowx - 1) ÷ (ngrid_x - 1) + 1, nelement_x)
      xstart = (xe - 1) * (ngrid_x - 1) + 1
      xend = xstart + ngrid_x - 1
      for icolx ∈ xstart:xend
        icol = (icolx - 1) * ny + irowy
        A[irow, icol] = rand(rng)
      end
    end

    # Add an 'y-derivative'
    for irowx in 1:nx, irowy in 1:ny
      irow = (irowx - 1) * ny + irowy
      ye = min((irowy - 1) ÷ (ngrid_y - 1) + 1, nelement_y)
      ystart = (ye - 1) * (ngrid_y - 1) + 1
      yend = ystart + ngrid_y - 1
      for icoly ∈ ystart:yend
        icol = (irowx - 1) * ny + icoly
        A[irow, icol] = rand(rng)

        # Make diagonal bigger to make sure matrix is not badly conditioned
        if icol == irow
          A[irow, icol] += 2.0
        end
      end
    end

    # Create one block per element
    local_blocks = Vector{Vector{Int}}()
    for xe ∈ 1:nelement_x, ye ∈ 1:nelement_y
      if xe == 1
        xstart = 1
        xblock_width = ngrid_x - 1
      else
        xstart = (xe - 1) * (ngrid_x - 1) + 2
        xblock_width = ngrid_x - 2
      end
      if xe == nelement_x
        xind = xstart:nx
      else
        xind = xstart:xstart+xblock_width-1
      end

      if ye == 1
        ystart = 1
        yblock_width = ngrid_y - 1
      else
        ystart = (ye - 1) * (ngrid_y - 1) + 2
        yblock_width = ngrid_y - 2
      end
      if ye == nelement_y
        yind = ystart:ny
      else
        yind = ystart:ystart+yblock_width-1
      end

      this_block = Int[]
      for ix ∈ xind, iy ∈ yind
        push!(this_block, (ix - 1) * ny + iy)
      end
      push!(local_blocks, this_block)
    end

    b = rand(rng, size(A, 1))
    check = A \ b
    Acf = CondensedFactorization(A, local_blocks; sparse_local_blocks=sparse)
    x = ldiv!(Acf, copy(b))
    @test isapprox(A * x, b; atol=fe_tol)
    @test isapprox(A * check, b; atol=fe_tol)
    @test isapprox(x, check; rtol=fe_tol)
  end
end
