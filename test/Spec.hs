import MiniJax

main :: IO ()
main = do
  -- Test that we can create operations
  print $ Add == Add  -- Should print: True
  print $ Add == Mul   -- Should print: False

  -- Test that we can create atoms
  let x = VarAtom "x"
  let three = LitAtom 3.0

  print x              -- Should print: VarAtom "x"
  print three          -- Should print: LitAtom 3.0

  -- Test equality
  print $ x == VarAtom "x"        -- Should print: True
  print $ x == VarAtom "y"        -- Should print: False
  print $ three == LitAtom 3.0    -- Should print: True