'use client'

import { useEffect } from 'react'
import { useQuery, queryOptions, useQueryClient } from '@tanstack/react-query'
import { useAppStore, useAppDispatch, useAppSelector } from '@/lib/hooks'
import {
  setSelectedImage,
  setEditorCanvasData,
  setSelectedMaskData,
  setMaskData,
  setMasks,
  setMaskIndex,
} from '@/lib/features/image-editor/imageEditorSlice'

import { setCollection } from '@/lib/features/control-panel/controlPanelSlice'
import { ScrollArea, ScrollBar } from './ui/scroll-area'

async function fetchImages(collection: string): Promise<MetadataEntry[]> {
  const res = await fetch(`http://127.0.0.1:8000/api/images`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ collection: collection }),
  })
  if (!res.ok) {
    throw new Error(`Failed to fetch metadata: ${res.status} ${res.statusText}`)
  }
  return res.json()
}

async function deleteImage(filename: string) {
  const res = await fetch(`http://localhost:8000/api/image/${filename}`, {
    method: 'DELETE',
    headers: {
      'Content-Type': 'application/json',
    },
  })
  if (!res.ok) {
    throw new Error(`Failed to delete image: ${res.status} ${res.statusText}`)
  }
}

export const ImageGallery = () => {
  const dispatch = useAppDispatch()
  const queryClient = useQueryClient()
  const {
    data: entries,
    error,
    isLoading,
    isError,
    refetch,
  } = useQuery(
    queryOptions({
      queryKey: ['images'],
      queryFn: async () => {
        const res = await fetch(`http://127.0.0.1:8000/api/images`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ collection: collection }),
        })

        if (!res.ok) {
          throw new Error(`Failed to fetch images: ${res.status} ${res.statusText}`)
        }
        return await res.json()
      },
      refetchOnWindowFocus: false,
      staleTime: Infinity,
    })
  )

  const collection = useAppSelector((state) => state.controlPanel.collection)

  // useEffect(() => {
  //   dispatch(setCollection('nsfw'))
  // }, [])

  useEffect(() => {
    ;(async () =>
      await queryClient.invalidateQueries({ queryKey: ['images'], refetchType: 'all' }))()
  }, [collection])

  const handleDelete = async (
    e: React.MouseEvent<HTMLImageElement, MouseEvent>,
    entry: MetadataEntry
  ) => {
    e.preventDefault()
    e.stopPropagation()
    try {
      await deleteImage(entry.filename)
      refetch()
    } catch (error) {
      console.error(error)
    }
  }

  if (isLoading) {
    return (
      <div className="text-center py-8">
        <p className="text-gray-400">Loading gallery...</p>
      </div>
    )
  }

  if (isError) {
    return (
      <div className="text-center py-8">
        <p className="text-red-500">Error: {error.message}</p>
        <button onClick={() => refetch()} className="mt-4 px-4 py-2 bg-red-600 text-white rounded">
          Retry
        </button>
      </div>
    )
  }

  if (!entries || entries.length === 0) {
    return (
      <div className="text-center py-8">
        <p className="text-gray-400">No images uploaded yet.</p>
      </div>
    )
  }

  return (
    <ScrollArea
      className={`flex flex-col z-10 max-w-[1200px] mx-auto px-4 py-8 h-[${entries.length * 400}px]`}
    >
      <div
        className={`relative grid grid-cols-1 sm:grid-cols-2 md:grid-cols-2 lg:grid-cols-3 gap-6 w-full h-[${entries.length * 400}px] m-0 overflow-y-scroll`}
      >
        {entries
          .filter((entry) => entry.collection == collection)
          .map((entry) => (
            <div
              key={entry.filename}
              className="relative z-10 bg-zinc-900 rounded-xl shadow-lg border border-zinc-700 overflow-hidden hover:scale-110 transition-transform duration-300 w-full h-auto aspect-square"
              onClick={() => {
                dispatch(setSelectedImage(entry))
                dispatch(setSelectedMaskData([]))
                dispatch(setMaskData([]))
                dispatch(setMaskIndex(0))
                dispatch(setMasks([]))
              }}
            >
              <img
                src={`data:image/${entry.filename.split('.').pop()};base64,${entry.image_data}`}
                alt={entry.original_filename}
                className="relative z-0 w-full h-48 aspect-auto object-center object-cover bg-gray-900 "
              />
              <button className="z-50 rounded-full bg-neutral-600 text-white p-1 absolute top-2 right-2">
                <img
                  src="/delete.svg"
                  alt="Delete"
                  className="w-4 h-4"
                  onClick={(e) => {
                    e.stopPropagation()
                    handleDelete(e, entry)
                  }}
                />
              </button>
              <div className="p-3">
                <p className="text-sm text-gray-200 truncate">{entry.label || 'No label'}</p>
                <p className="text-xs text-gray-500">
                  {new Date(entry.timestamp).toLocaleString()}
                </p>
              </div>
            </div>
          ))}
        <ScrollBar orientation="vertical" />
      </div>
    </ScrollArea>
  )
}
