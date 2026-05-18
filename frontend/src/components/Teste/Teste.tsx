import { ReactNode } from "react";

interface ITeste {
  children: ReactNode
}

function Teste({children}: ITeste) {
  return <div>{children}</div>
}

export default Teste