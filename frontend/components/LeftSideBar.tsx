import { useState } from 'react'
import { ImageUploadForm } from './ImageUploadForm'
import { ImageGallery } from './ImageGallery'

export const LeftSideBar = () => {
  const [images, setImages] = useState<string[]>([])

  return (
    <aside className="w-1/5 min-w-[200px] bg-neutral-900 border-r border-neutral-800 p-2 overflow-y-auto">
      <h2 className="text-xs font-semibold text-neutral-400 mb-2">Upload Image</h2>
      <ImageUploadForm />
      <h2 className="text-xs font-semibold text-neutral-400 mt-6 mb-2">Gallery</h2>
      <ImageGallery />
    </aside>
  )
}
