'use client'

import { useState } from 'react'
import { cn } from '@/lib/utils'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { toast } from 'sonner'

export const ImageUploadForm = ({ onUpload }: { onUpload?: () => void }) => {
  const queryClient = useQueryClient()

  const uploadMutation = useMutation({
    mutationFn: async (formData: FormData) => {
      return fetch('/api/upload', {
        method: 'POST',
        body: formData,
      })
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['images'] })
    },
  })

  const [image, setImage] = useState<File | null>(null)
  const [label, setLabel] = useState('')
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [isDragging, setIsDragging] = useState(false)
  const [uploading, setUploading] = useState(false)

  const handleDrop = (event: React.DragEvent<HTMLLabelElement>) => {
    event.preventDefault()
    event.stopPropagation()
    setIsDragging(false)
    const file = event.dataTransfer.files[0]
    if (file && file.type.startsWith('image/')) {
      setImage(file)
      setPreviewUrl(URL.createObjectURL(file))
    }
  }

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file && file.type.startsWith('image/')) {
      setImage(file)
      setPreviewUrl(URL.createObjectURL(file))
    }
  }

  const handleUpload = async () => {
    if (!image) return

    const formData = new FormData()
    formData.append('file', image)
    formData.append('label', label)

    // This triggers the mutation (and its onSuccess, which invalidates the images query)
    uploadMutation.mutate(formData, {
      onSuccess: () => {
        if (typeof onUpload === 'function') {
          onUpload()
        }
        alert('Upload successful!')
        setImage(null)
        setPreviewUrl(null)
        setLabel('')
        setUploading(false)
      },
      onError: (error: any) => {
        setUploading(false)
        alert('Upload failed.')
        console.error(error)
      },
    })

    setUploading(true) // Optional: If you want to show loading UI while uploadMutation is pending
  }

  return (
    <form
      onSubmit={(e) => {
        e.preventDefault()
        handleUpload()
      }}
      className="space-y-4"
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
          <span className="text-gray-500">Drag & Drop or Click to Select an Image</span>
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
        placeholder="Enter label (optional)"
        className="w-full p-2 border rounded-md"
        value={label}
        onChange={(e) => setLabel(e.target.value)}
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
