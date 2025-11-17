module ContextSpec (spec) where

import Test.Hspec
import MiniJax
import Control.Monad.Reader

-- A mock interpreter that always returns 42.0
data MockInterpreter = MockInterpreter

instance Interpreter MockInterpreter where
  interpretOp _ _ _ = 42.0

mockInterpreter :: MockInterpreter
mockInterpreter = MockInterpreter

spec :: Spec
spec = do
  describe "Interpreter context switching" $ do
    it "should use default EvalInterpreter" $ do
      -- By default, add and mul should do normal arithmetic
      runJax (add 2.0 3.0) `shouldBe` 5.0
      runJax (mul 2.0 3.0) `shouldBe` 6.0

    it "should switch to different interpreter" $ do
      -- When we switch context, add and mul should use the new interpreter
      let computation = add 2.0 3.0
      runWithInterpreter mockInterpreter computation `shouldBe` 42.0

    it "should compose operations in monadic context" $ do
      let foo x = do
            y <- add x 3.0
            mul x y
      runJax (foo 2.0) `shouldBe` 10.0

    it "should handle nested interpreter switches" $ do
      let computation = do
            x <- add 1.0 2.0  -- Uses outer interpreter
            local (const mockInterpreter) $ do
              y <- add 3.0 4.0  -- Uses MockInterpreter, returns 42.0
              return (x, y)
      runJax computation `shouldBe` (3.0, 42.0)