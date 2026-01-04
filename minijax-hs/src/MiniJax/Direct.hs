{-# LANGUAGE BangPatterns #-}

-- | Direct-style MiniJax implementation, mirroring OCaml\/Python.
--
-- = Design: Reader Monad with Value Nesting
--
-- This implementation mirrors the OCaml\/Python style more closely than the
-- tagless final approach. Key design choices:
--
-- 1. __One Value type__: All interpreters work with the same 'Value' type,
--    which can be @VFloat@, @VDual@, or @VAtom@. This enables natural nesting
--    of dual numbers for higher-order AD.
--
-- 2. __Interpreter as record__: The 'Interpreter' type is a simple record
--    with a function for each operation, rather than a type class.
--
-- 3. __Reader monad__: Operations run in @ReaderT Interpreter@ to access
--    the current interpreter, similar to implicit threading in the Python
--    version but pure.
--
-- = Perturbation Confusion
--
-- Each 'Dual' number carries an @Int@ tag identifying its creating interpreter.
-- The 'lift' operation checks this tag: if a Dual has a /different/ tag, it's
-- treated as a constant (tangent = 0). This prevents perturbation confusion
-- in higher-order differentiation.
--
-- = Comparison to Tagless Final
--
-- Unlike 'MiniJax.Tagless', this approach:
--
-- * Uses runtime dispatch (slower, more flexible)
-- * Supports natural higher-order AD via 'Value' nesting
-- * Has simpler types (no associated type families)
--
-- For simple forward-mode AD, the tagless final version is more type-safe.
-- For higher-order AD and more complex transformations, this direct style
-- may be easier to extend.
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
import Control.Monad.State.Strict
import qualified Data.Map.Strict as Map

data JaxState = JaxState
  { nextTag :: !Int
  , nextVar :: !Int
  , equations :: [Equation]
  }

initialJaxState :: JaxState
initialJaxState = JaxState
  { nextTag = 0
  , nextVar = 0
  , equations = []
  }

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
  { interpOp :: Op -> Value -> Value -> Jax Value
  }

-- | Computations run in a Reader over the current interpreter
type Jax = ReaderT Interpreter (State JaxState)

-- | Run a Jax computation with a given interpreter
runJax :: Interpreter -> Jax a -> a
runJax interp m = evalState (runReaderT m interp) initialJaxState

-- | Run a computation with a different interpreter
withInterpreter :: Interpreter -> Jax a -> Jax a
withInterpreter = local . const

-- | Binary operations dispatch to current interpreter
add, mul :: Value -> Value -> Jax Value
add x y = do
  i <- ask
  interpOp i Add x y
mul x y = do
  i <- ask
  interpOp i Mul x y

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
    evalOp Add (VFloat x) (VFloat y) = return (VFloat (x + y))
    evalOp Mul (VFloat x) (VFloat y) = return (VFloat (x * y))
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
    prevOp :: Op -> Value -> Value -> Jax Value
    prevOp = interpOp prev

    jvpOp :: Op -> Value -> Value -> Jax Value
    jvpOp op x y = do
      let dx = liftToDual x
          dy = liftToDual y
          px = primalV dx
          py = primalV dy
          tx = tangentV dx
          ty = tangentV dy
      case op of
        Add -> do
          p <- prevOp Add px py
          t <- prevOp Add tx ty
          return (VDual (Dual myTag p t))
        Mul -> do
          p <- prevOp Mul px py
          t1 <- prevOp Mul px ty
          t2 <- prevOp Mul tx py
          t <- prevOp Add t1 t2
          return (VDual (Dual myTag p t))

-- | Compute JVP of a function
jvp :: (Value -> Jax Value) -> Value -> Value -> Jax (Value, Value)
jvp f primal tangent = do
  prev <- ask
  myTag <- freshTag
  let jvpInterp = makeJvpInterp myTag prev
      dualIn = VDual (Dual myTag primal tangent)
  result <- withInterpreter jvpInterp (f dualIn)
  let dualOut = case result of
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
runStaging :: Int -> Jax Value -> (Value, [Equation])
runStaging startCounter m =
  let toAtom :: Value -> Atom
      toAtom (VAtom a)  = a
      toAtom (VFloat x) = LitAtom x
      toAtom (VDual _)  = error "cannot stage dual number"

      stageOp :: Op -> Value -> Value -> Jax Value
      stageOp op x y = do
        let !xAtom = toAtom x
            !yAtom = toAtom y
        v <- freshVar
        modify' (\st' -> st' { equations = equations st' ++ [Equation v op [xAtom, yAtom]] })
        return (VAtom (VarAtom v))

      interp = Interpreter stageOp
      initial = initialJaxState { nextVar = startCounter, equations = [] }
      (result, st) = runState (runReaderT m interp) initial
  in (result, equations st)

-- | Evaluate a Jaxpr
evalJaxpr :: Jaxpr -> [Value] -> Jax Value
evalJaxpr jaxpr args = do
  let env0 = Map.fromList $ zip (jaxprParams jaxpr) args

      evalAtom :: Map.Map String Value -> Atom -> Value
      evalAtom _ (LitAtom x)  = VFloat x
      evalAtom env (VarAtom v) = env Map.! v

  interp <- ask
  let go env [] = return (evalAtom env (jaxprReturn jaxpr))
      go env (Equation v op atoms : rest) = do
        case map (evalAtom env) atoms of
          [a1, a2] -> do
            result <- interpOp interp op a1 a2
            go (Map.insert v result env) rest
          _ -> error "evalJaxpr: expected two arguments"

  go env0 (jaxprEqns jaxpr)

freshTag :: Jax Int
freshTag = do
  st <- get
  let next = nextTag st + 1
  put st { nextTag = next }
  return next

freshVar :: Jax String
freshVar = do
  st <- get
  let next = nextVar st + 1
      v = "v_" ++ show next
  put st { nextVar = next }
  return v
