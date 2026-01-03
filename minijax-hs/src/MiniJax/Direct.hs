{-# LANGUAGE BangPatterns #-}

-- | Direct-style MiniJax implementation, mirroring OCaml/Python.
--
-- This is simpler than the tagless final version because:
-- 1. One Value type for all interpreters (natural nesting)
-- 2. Interpreter is just a record of functions
-- 3. Higher-order AD falls out naturally
module MiniJax.Direct
  ( -- * Types
    Op(..)
  , Value(..)
  , Dual(..)
  , Interpreter(..)
  , Atom(..)
  , Equation(..)
  , Jaxpr(..)
    -- * Running computations
  , Jax
  , runJax
  , withInterpreter
    -- * Operations
  , add
  , mul
  , lit
    -- * Interpreters
  , evalInterp
  , makeJvpInterp
    -- * Differentiation
  , jvp
  , derivative
  , nthDerivative
    -- * Staging
  , buildJaxpr
  , evalJaxpr
    -- * Utilities
  , valueToDouble
  ) where

import Control.Monad.Reader
import Data.IORef
import System.IO.Unsafe (unsafePerformIO)
import qualified Data.Map.Strict as Map

-- Use a global counter for unique tags (safer than Unique with unsafePerformIO)
{-# NOINLINE tagCounter #-}
tagCounter :: IORef Int
tagCounter = unsafePerformIO $ newIORef 0

{-# NOINLINE freshTag #-}
freshTag :: IO Int
freshTag = atomicModifyIORef' tagCounter (\n -> (n + 1, n + 1))

-- | Primitive operations
data Op = Add | Mul
  deriving (Show, Eq)

-- | Values that flow through interpreters
data Value
  = VFloat !Double
  | VDual !Dual
  | VAtom !Atom
  deriving (Show, Eq)

-- | Dual number with interpreter tag for perturbation confusion
data Dual = Dual
  { dualTag  :: !Int       -- ^ Which interpreter created this
  , primalV  :: !Value
  , tangentV :: !Value
  }

instance Show Dual where
  show (Dual _ p t) = "Dual { primal = " ++ show p ++ ", tangent = " ++ show t ++ " }"

instance Eq Dual where
  d1 == d2 = dualTag d1 == dualTag d2 && primalV d1 == primalV d2 && tangentV d1 == tangentV d2

-- | Atoms for staging
data Atom = VarAtom String | LitAtom Double
  deriving (Show, Eq)

-- | An equation in the IR
data Equation = Equation
  { eqVar  :: String
  , eqOp   :: Op
  , eqArgs :: [Atom]
  } deriving (Show, Eq)

-- | Jaxpr IR
data Jaxpr = Jaxpr
  { jaxprParams :: [String]
  , jaxprEqns   :: [Equation]
  , jaxprReturn :: Atom
  } deriving (Show, Eq)

-- | Interpreter is a record of how to handle each operation
data Interpreter = Interpreter
  { interpOp :: Op -> Value -> Value -> Value
  }

-- | Computations run in a Reader over the current interpreter
type Jax = Reader Interpreter

-- | Run a Jax computation with a given interpreter
runJax :: Interpreter -> Jax a -> a
runJax interp m = runReader m interp

-- | Run a computation with a different interpreter
withInterpreter :: Interpreter -> Jax a -> Jax a
withInterpreter = local . const

-- | Binary operations dispatch to current interpreter
add, mul :: Value -> Value -> Jax Value
add x y = asks (\i -> interpOp i Add x y)
mul x y = asks (\i -> interpOp i Mul x y)

-- | Lift a literal (interpreter-independent)
lit :: Double -> Value
lit = VFloat

-- | Create zero with same structure
zeroLike :: Value -> Value
zeroLike (VFloat _) = VFloat 0
zeroLike (VAtom _)  = VAtom (LitAtom 0)
zeroLike (VDual d)  = VDual $ Dual (dualTag d) (zeroLike (primalV d)) (zeroLike (tangentV d))

-- | Extract Double from (possibly nested) value
valueToDouble :: Value -> Double
valueToDouble (VFloat x) = x
valueToDouble (VDual d)  = valueToDouble (primalV d)
valueToDouble (VAtom _)  = error "valueToDouble: cannot convert atom"

--------------------------------------------------------------------------------
-- Eval Interpreter
--------------------------------------------------------------------------------

-- | Evaluation interpreter - just does arithmetic
evalInterp :: Interpreter
evalInterp = Interpreter evalOp
  where
    evalOp Add (VFloat x) (VFloat y) = VFloat (x + y)
    evalOp Mul (VFloat x) (VFloat y) = VFloat (x * y)
    evalOp _ _ _ = error "evalInterp: expected VFloat arguments"

--------------------------------------------------------------------------------
-- JVP Interpreter
--------------------------------------------------------------------------------

-- | Create a JVP interpreter with a fresh tag
makeJvpInterp :: Int -> Interpreter -> Interpreter
makeJvpInterp myTag prev = Interpreter jvpOp
  where
    -- Lift a value to a Dual. If it's already a Dual with our tag, keep it.
    -- Otherwise treat it as constant (tangent = 0).
    liftToDual :: Value -> Dual
    liftToDual (VDual d) | dualTag d == myTag = d
    liftToDual v = Dual myTag v (zeroLike v)

    -- Run an operation in the previous interpreter
    prevOp :: Op -> Value -> Value -> Value
    prevOp = interpOp prev

    jvpOp :: Op -> Value -> Value -> Value
    jvpOp op x y =
      let dx = liftToDual x
          dy = liftToDual y
          px = primalV dx
          py = primalV dy
          tx = tangentV dx
          ty = tangentV dy
      in case op of
        Add ->
          let p = prevOp Add px py
              t = prevOp Add tx ty
          in VDual (Dual myTag p t)
        Mul ->
          let p  = prevOp Mul px py
              t1 = prevOp Mul px ty
              t2 = prevOp Mul tx py
              t  = prevOp Add t1 t2
          in VDual (Dual myTag p t)

-- | Compute JVP of a function
{-# NOINLINE jvp #-}
jvp :: (Value -> Jax Value) -> Value -> Value -> Jax (Value, Value)
jvp f primal tangent = do
  prev <- ask
  -- Wrap entire computation in unsafePerformIO to ensure freshTag runs each time
  -- The $! forces evaluation to trigger the IO action
  return $! unsafePerformIO $ do
    myTag <- freshTag
    let jvpInterp = makeJvpInterp myTag prev
        dualIn = VDual (Dual myTag primal tangent)
        result = runJax jvpInterp (f dualIn)
        -- Lift result in case f returned a constant
        dualOut = case result of
          VDual d | dualTag d == myTag -> d
          v -> Dual myTag v (zeroLike v)
    return (primalV dualOut, tangentV dualOut)

-- | Compute derivative of f at x
derivative :: (Value -> Jax Value) -> Double -> Jax Value
derivative f x = snd <$> jvp f (VFloat x) (VFloat 1)

-- | Compute nth derivative of f at x
-- Higher-order AD works naturally because Value can nest!
nthDerivative :: Int -> (Value -> Jax Value) -> Double -> Jax Value
nthDerivative n f x = nthDerivativeValue n f (VFloat x)

-- | Internal helper that keeps Value structure through recursion
nthDerivativeValue :: Int -> (Value -> Jax Value) -> Value -> Jax Value
nthDerivativeValue n f v
  | n < 0     = error "nthDerivative: negative order"
  | n == 0    = f v
  | otherwise = do
      -- Differentiate "the function that computes the (n-1)th derivative"
      -- Key: pass u directly, NOT valueToDouble u, to preserve dual structure
      let f' u = nthDerivativeValue (n - 1) f u
      (_, t) <- jvp f' v (VFloat 1)
      return t

--------------------------------------------------------------------------------
-- Stage Interpreter
--------------------------------------------------------------------------------

-- | Build a Jaxpr from a function using a simple state-passing approach
buildJaxpr :: Int -> ([Value] -> Jax Value) -> Jaxpr
buildJaxpr numArgs f =
  let params = ["v_" ++ show i | i <- [1..numArgs]]
      paramVals = map (VAtom . VarAtom) params
      -- Use a staging interpreter that returns equations along with result
      (result, eqns) = runStaging numArgs (f paramVals)
      retAtom = case result of
        VAtom a  -> a
        VFloat x -> LitAtom x
        VDual _  -> error "cannot stage dual"
  in Jaxpr params eqns retAtom

-- | Run a Jax computation with staging, returning result and equations
{-# NOINLINE runStaging #-}
runStaging :: Int -> Jax Value -> (Value, [Equation])
runStaging startCounter m = unsafePerformIO $ do
  counterRef <- newIORef startCounter
  eqnsRef <- newIORef []

  let toAtom :: Value -> Atom
      toAtom (VAtom a)  = a
      toAtom (VFloat x) = LitAtom x
      toAtom (VDual _)  = error "cannot stage dual number"

      -- CRITICAL: Force evaluation of arguments BEFORE incrementing counter
      -- This ensures nested unsafePerformIO from argument evaluation happens first
      stageOp :: Op -> Value -> Value -> Value
      stageOp op x y =
        let !xAtom = toAtom x  -- Force x first (may trigger nested staging)
            !yAtom = toAtom y  -- Force y second (may trigger nested staging)
        in unsafePerformIO $ do
          n <- atomicModifyIORef' counterRef (\c -> (c + 1, c + 1))
          let v = "v_" ++ show n
              eqn = Equation v op [xAtom, yAtom]
          atomicModifyIORef' eqnsRef (\es -> (es ++ [eqn], ()))
          return $! VAtom (VarAtom v)

      interp = Interpreter stageOp

  let !result = runReader m interp
  eqns <- readIORef eqnsRef
  return (result, eqns)

-- | Evaluate a Jaxpr
evalJaxpr :: Jaxpr -> [Value] -> Jax Value
evalJaxpr jaxpr args = do
  let env0 = Map.fromList $ zip (jaxprParams jaxpr) args

      evalAtom :: Map.Map String Value -> Atom -> Value
      evalAtom _ (LitAtom x)  = VFloat x
      evalAtom env (VarAtom v) = env Map.! v

  interp <- ask
  let go env [] = evalAtom env (jaxprReturn jaxpr)
      go env (Equation v op atoms : rest) =
        let [a1, a2] = map (evalAtom env) atoms
            result = interpOp interp op a1 a2
        in go (Map.insert v result env) rest

  return $ go env0 (jaxprEqns jaxpr)
