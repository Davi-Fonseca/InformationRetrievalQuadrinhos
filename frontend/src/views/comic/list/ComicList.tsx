'use client'

import ComicCard from "./ComicCard"
import { ComicDTO } from "../Comic.types"
import { TextInput } from "flowbite-react";
import { useEffect, useState } from "react";

interface ComicListProps {
    comics?: ComicDTO[]
}

function ComicList({ comics: initialComics = [] }: ComicListProps) {
    const [searchInput, setSearchInput] = useState('')
    const [searchString, setSearchString] = useState('')
    const [comics, setComics] = useState<ComicDTO[]>(initialComics)
    const [loading, setLoading] = useState(false)

    useEffect(() => {
        // Only fetch if searchString changes (i.e., Enter pressed)
        setLoading(true)
        fetch(`${process.env.NEXT_PUBLIC_API}/search?q=${encodeURIComponent(searchString)}`)
            .then(res => res.json())
            .then(json => setComics(json.data))
            .finally(() => setLoading(false))
    }, [searchString])

    const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        setSearchInput(e.target.value)
    }

    const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
        if (e.key === 'Enter') {
            setSearchString(searchInput)
        }
    }

    return (
        <div>
            <div className="flex justify-between mb-2 items-center">
                <h1 className="text-2xl font-bold">Lista de Quadrinhos</h1>
                <TextInput
                    onChange={handleInputChange}
                    onKeyDown={handleKeyDown}
                    value={searchInput}
                    className="w-80"
                    type="text"
                    placeholder="Busca..."
                />
            </div>
            {loading && <div>Carregando...</div>}
            <div className="grid sm:grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                {comics.map(p => <ComicCard key={p.id} comic={p}/>)}
            </div>
        </div>
    )
}

export default ComicList