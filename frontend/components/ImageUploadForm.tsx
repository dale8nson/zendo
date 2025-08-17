'use client'

import { useState, useRef } from 'react'
import { cn } from '@/lib/utils'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { toast } from 'sonner'
import { setCollection } from '@/lib/features/control-panel/controlPanelSlice'
import { useAppSelector, useAppDispatch } from '@/lib/hooks'

export const ImageUploadForm = ({ onUpload }: { onUpload?: () => void }) => {
  const collection = useAppSelector((state) => state.controlPanel.collection)

  const dispatch = useAppDispatch()

  const queryClient = useQueryClient()

  const uploadMutation = useMutation({
    mutationFn: async (formData: FormData) => {
      return fetch('http://127.0.0.1:8000/api/upload', {
        method: 'POST',
        body: formData,
      })
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['images'], refetchType: 'all' })
    },
  })

  const [image, setImage] = useState<File | null>(null)

  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [isDragging, setIsDragging] = useState(false)
  const [uploading, setUploading] = useState(false)

  const debounce = useRef(false)

  const handleDrop = (event: React.DragEvent<HTMLLabelElement>) => {
    event.preventDefault()
    event.stopPropagation()
    setIsDragging(false)
    const file = event.dataTransfer.files[0]
    if (file && file.type.startsWith('image/')) {
      setImage(file)
      const url = URL.createObjectURL(file)
      setPreviewUrl(url)
    }
  }

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file && file.type.startsWith('image/')) {
      setImage(file)
      const url = URL.createObjectURL(file)
      console.log(`url:${url.slice(0, 29)}...`)
      setPreviewUrl(url)
    }
  }

  const handleUpload = async () => {
    if (!image) return

    const formData = new FormData()
    formData.append('file', image)
    formData.append('collection', collection)

    // This triggers the mutation (and its onSuccess, which invalidates the images query)
    uploadMutation.mutate(formData, {
      onSuccess: () => {
        if (typeof onUpload === 'function') {
          onUpload()
        }
        toast.success('Upload successful!')
        setImage(null)
        setPreviewUrl(null)
        setUploading(false)
        queryClient.invalidateQueries({ queryKey: ['images'], refetchType: 'all' })
      },
      onError: (error: any) => {
        setUploading(false)
        alert('Upload failed.')
        console.error(error)
      },
    })

    setUploading(true)
  }

  return (
    <form
      onSubmit={(e) => {
        e.preventDefault()
        handleUpload()
      }}
      className="space-y-4 sticky top-0 z-50 bg-neutral-900"
    >
      <label
        htmlFor="file"
        onDrop={handleDrop}
        onDragOver={(e) => {
          e.preventDefault()
          setIsDragging(true)
        }}
        onDragLeave={() => setIsDragging(false)}
        className={cn(
          'w-full h-48 border-2 border-dashed rounded-md flex items-center justify-center cursor-pointer transition-all',
          isDragging ? 'border-blue-500 bg-blue-50' : 'border-gray-300'
        )}
      >
        {previewUrl ? (
          <img src={previewUrl} alt="Preview" className="h-full object-contain" />
        ) : (
          <span className="text-gray-500 p-2">Drag & Drop or Click to Select an Image</span>
        )}
        <input
          type="file"
          id="file"
          name="file"
          accept="image/*"
          className="hidden"
          onChange={handleFileChange}
        />
      </label>
      <input
        type="text"
        placeholder="default"
        defaultValue="default"
        className="w-full p-2 border rounded-md"
        onKeyDown={(e) => {
          if (e.key == 'Enter') {
            dispatch(setCollection(e.target.value))
          }
        }}
      />
      <button
        type="submit"
        className="w-full py-2 px-4 rounded-md text-white bg-gradient-to-br from-gray-800 to-black hover:from-black hover:to-gray-900"
        disabled={uploadMutation.isPending}
      >
        {uploadMutation.isPending ? 'Uploading...' : 'Upload'}
      </button>
    </form>
  )
}
