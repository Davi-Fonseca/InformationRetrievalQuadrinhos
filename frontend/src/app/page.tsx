import ComicList from "@/views/comic/list/ComicList"
import { ComicDTO } from "@/views/comic/Comic.types"
import fs from "fs"
import path from "path"
import Papa from "papaparse"

async function Home() {
  // const res = await fetch(`${process.env.NEXT_PUBLIC_DOCKER_API}/comics`) // usando docker no front com backend antigo
  // const res = await fetch(`${process.env.NEXT_PUBLIC_API}/comics`) // sem usar docker no front

  // const comics: ComicDTO[] = await res.json()

  // const csvPath = path.join(process.cwd(), "data", "marvel.csv")
  // const csvData = fs.readFileSync(csvPath, "utf8")
  // const { data } = Papa.parse<ComicDTO>(csvData, { header: true })
  // const comics = data.filter(Boolean) // Remove empty rows if any

  return <ComicList comics={[]}/>
}

export default Home