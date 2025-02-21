using StableRNGs
using Test

using MPIStaticCondensation

function ldiv_wrapper(A, b)
  x = similar(b)
  return ldiv!(x, A, b)
end

@testset "CondensedFactorization" begin
  @testset "$label sparse=$sparse" for (label, solvefunc) in (
                                               ("static_condensed_solve", static_condensed_solve),
                                               ("ldiv!", ldiv_wrapper),
                                              ),
                                       sparse in (false, true)
    @testset "($L,$C)" for (L, C, tol) in (
                                           (2, 1, 1.0e-14),
                                           (3, 2, 1.0e-14),
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
      x = solvefunc(Acf, b)
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
      x = solvefunc(Acf, b)
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
      x = solvefunc(Acf, b)
      @test isapprox(A * x, b; atol=tol)
      @test isapprox(A * check, b; atol=tol)
      @test isapprox(x, check; rtol=tol)
    end
  end

  @testset "$label sparse=$sparse" for (label, solvefunc) in (
                                               ("static_condensed_solve", static_condensed_solve),
                                               ("ldiv!", ldiv_wrapper),
                                              ),
                                       sparse in (false, true)
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
      x = solvefunc(Acf, b)
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
        x = solvefunc(Acf, b)
        @test isapprox(A * x, b; atol=tol)
        @test isapprox(A * check, b; atol=tol)
        @test isapprox(x, check; rtol=tol)
      end
    end
  end
end
