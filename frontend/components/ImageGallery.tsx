'use client'

import { useQuery, queryOptions } from '@tanstack/react-query'
import { useAppStore, useAppDispatch, useAppSelector } from '@/lib/hooks'
import { setSelectedImage } from '@/lib/features/image-editor/imageEditorSlice'

export interface MetadataEntry {
  id: number
  filename: string
  original_filename: string
  label: string | null
  prediction: string | null
  timestamp: string
  image_data: string
}

async function fetchImages(): Promise<MetadataEntry[]> {
  // const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api'
  const res = await fetch(`/api/images`, {
    headers: { 'Access-Control-Allow-Origin': '*' },
  })
  if (!res.ok) {
    throw new Error(`Failed to fetch metadata: ${res.status} ${res.statusText}`)
  }
  return res.json()
}

export const ImageGallery = () => {
  const dispatch = useAppDispatch()

  const {
    data: entries,
    error,
    isLoading,
    isError,
    refetch,
  } = useQuery(
    queryOptions({
      queryKey: ['images'],
      queryFn: fetchImages,
      staleTime: 1000 * 60,
      refetchOnWindowFocus: true,
    })
  )

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
    <div className="max-w-[1200px] mx-auto px-4 py-8">
      <h2 className="text-2xl font-semibold mb-4 text-center text-white">Uploaded Images</h2>
      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-2 gap-6 w-full">
        {entries.map((entry) => (
          <div
            key={entry.filename}
            className="bg-zinc-900 rounded-xl shadow-lg border border-zinc-700 overflow-hidden hover:scale-110 transition-transform duration-300 w-full h-auto"
            onClick={() => dispatch(setSelectedImage(entry))}
          >
            <img
              src={`data:image/${entry.filename.split('.').pop()};base64,${entry.image_data}`}
              alt={entry.label || entry.original_filename}
              className="w-full h-48 aspect-auto object-cover bg-gray-900 object-top"
            />
            <div className="p-3">
              <p className="text-sm text-gray-200 truncate">{entry.label || 'No label'}</p>
              <p className="text-xs text-gray-500">{new Date(entry.timestamp).toLocaleString()}</p>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
