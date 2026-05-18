import ComicDetails from "@/views/comic/item/ComicDetails"

import { ComicDTO } from "@/views/comic/Comic.types"
import fs from "fs"
import path from "path"
import Papa from "papaparse"

interface ComicPageProps {
  params: Promise<{
    id: string
  }>
}

async function ComicPage({params}: ComicPageProps) {
  const { id } = await params

  // const resp = await fetch(`${process.env.NEXT_PUBLIC_DOCKER_API}/comics/${id}`)
  // const resp = await fetch(`${process.env.NEXT_PUBLIC_API}/comics/${id}`) // sem usar docker no front
  
  // const comic = await resp.json()

  const csvPath = path.join(process.cwd(), "data", "marvel.csv")
  const csvData = fs.readFileSync(csvPath, "utf8")
  const { data } = Papa.parse<ComicDTO>(csvData, { header: true })

  const comic = data.find(c => c.id === id)

  if (!comic) {
    return <div>Comic not found</div>
  }

  return (
    <ComicDetails comic={comic}></ComicDetails>
  )
}

export default ComicPage