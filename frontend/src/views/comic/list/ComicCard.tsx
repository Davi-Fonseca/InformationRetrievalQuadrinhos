"use client"

import { Card } from "flowbite-react";
import { memo } from "react";
import { ComicDTO } from "../Comic.types";
import Link from "next/link";

interface ComicCardProps {
  comic: ComicDTO;
}

function ComicCard({ comic }: ComicCardProps) {

  return (
    <Card className="max-w-sm">
      <div className="flex flex-col justify-between h-full">
        <h5 className="text-2xl font-bold tracking-tight text-gray-900 dark:text-white">
          <Link href={`/comic/${comic.id}`}>{comic.issue_title}</Link>
        </h5>
        <p className="font-normal text-gray-700 dark:text-gray-400">
          {comic.issue_description}
        </p>
      </div>
    </Card>
  );
}

export default memo(ComicCard)
