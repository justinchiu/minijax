module DirectSpec (spec) where

import Test.Hspec
import MiniJax.Direct

-- | foo(x) = x * (x + 3) = x^2 + 3x
foo :: Value -> Jax Value
foo x = do
  three <- return $ lit 3
  y <- add x three
  mul x y

-- | Perturbation confusion test: f(x) = x * derivative(g, 0) where g(y) = x
-- Since g is constant in y, derivative should be 0, so f'(0) = 0
perturbationTest :: Value -> Jax Value
perturbationTest x = do
  let g _ = return x  -- g ignores its argument, returns x
  shouldBeZero <- derivative g 0
  mul x shouldBeZero

spec :: Spec
spec = do
  describe "Direct-style MiniJax" $ do

    describe "Evaluation" $ do
      it "should evaluate add" $ do
        runJax evalInterp (add (lit 2) (lit 3)) `shouldBe` VFloat 5

      it "should evaluate mul" $ do
        runJax evalInterp (mul (lit 2) (lit 3)) `shouldBe` VFloat 6

      it "should evaluate foo(2) = 10" $ do
        runJax evalInterp (foo (lit 2)) `shouldBe` VFloat 10

    describe "JVP (forward-mode AD)" $ do
      it "should differentiate add" $ do
        let (p, t) = runJax evalInterp $ jvp (\x -> add x (lit 3)) (lit 2) (lit 1)
        p `shouldBe` VFloat 5
        t `shouldBe` VFloat 1

      it "should differentiate mul" $ do
        let (p, t) = runJax evalInterp $ jvp (\x -> mul x (lit 3)) (lit 2) (lit 1)
        p `shouldBe` VFloat 6
        t `shouldBe` VFloat 3

      it "should differentiate foo: foo'(2) = 7" $ do
        let (p, t) = runJax evalInterp $ jvp foo (lit 2) (lit 1)
        p `shouldBe` VFloat 10
        t `shouldBe` VFloat 7

    describe "Higher-order AD" $ do
      -- foo(x) = x^2 + 3x
      -- foo'(x) = 2x + 3
      -- foo''(x) = 2
      -- foo'''(x) = 0

      it "0th derivative of foo at x=2 is 10" $ do
        valueToDouble (runJax evalInterp $ nthDerivative 0 foo 2) `shouldBe` 10

      it "1st derivative of foo at x=2 is 7" $ do
        valueToDouble (runJax evalInterp $ nthDerivative 1 foo 2) `shouldBe` 7

      it "2nd derivative of foo at x=2 is 2" $ do
        valueToDouble (runJax evalInterp $ nthDerivative 2 foo 2) `shouldBe` 2

      it "3rd derivative of foo at x=2 is 0" $ do
        valueToDouble (runJax evalInterp $ nthDerivative 3 foo 2) `shouldBe` 0

      it "4th derivative of foo at x=2 is 0" $ do
        valueToDouble (runJax evalInterp $ nthDerivative 4 foo 2) `shouldBe` 0

    describe "Perturbation confusion" $ do
      it "should correctly handle nested differentiation" $ do
        let (_, t) = runJax evalInterp $ jvp perturbationTest (lit 0) (lit 1)
        t `shouldBe` VFloat 0

    describe "Staging" $ do
      it "should stage foo into a Jaxpr" $ do
        let jaxpr = buildJaxpr 1 $ \[x] -> foo x
        jaxprParams jaxpr `shouldBe` ["v_1"]
        jaxprEqns jaxpr `shouldBe`
          [ Equation "v_2" Add [VarAtom "v_1", LitAtom 3]
          , Equation "v_3" Mul [VarAtom "v_1", VarAtom "v_2"]
          ]
        jaxprReturn jaxpr `shouldBe` VarAtom "v_3"

      it "should evaluate a Jaxpr" $ do
        let jaxpr = buildJaxpr 1 $ \[x] -> foo x
            result = runJax evalInterp $ evalJaxpr jaxpr [lit 2]
        result `shouldBe` VFloat 10

      it "should differentiate through evalJaxpr" $ do
        let jaxpr = buildJaxpr 1 $ \[x] -> foo x
            (p, t) = runJax evalInterp $ jvp (\x -> evalJaxpr jaxpr [x]) (lit 2) (lit 1)
        p `shouldBe` VFloat 10
        t `shouldBe` VFloat 7
