import { ComicDTO } from "../Comic.types";

interface ComicDetailsProps {
  comic: ComicDTO
}

function ComicDetails({ comic }: ComicDetailsProps) {
  return (
    <div>
      <h1 className="text-2xl font-bold mb-2">{comic.issue_title}</h1>
      <div>{comic.issue_description}</div>
      <div><span className="font-bold">Nome do volume:</span> {comic.comic_name}</div>
      <div><span className="font-bold">Escritor:</span> {comic.writer}</div>
      <div><span className="font-bold">Desenhista:</span> {comic.penciler}</div>
      <div><span className="font-bold">Artista da Capa:</span> {comic.cover_artist}</div>
    </div>
  )
}

export default ComicDetails