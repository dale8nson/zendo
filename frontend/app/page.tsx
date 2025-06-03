import { ImageUploadForm } from '@/components/ImageUploadForm'
import { ImageGallery } from '@/components/ImageGallery'

export default function Home() {
  return (
    <main className="min-h-screen p-4 bg-gradient-to-b from-zinc-900 via-black to-zinc-800 text-white">
      <h1 className="text-3xl font-bold">ZendoAI</h1>
      <ImageUploadForm />
      <ImageGallery />
      {/* Editor / Prediction Display */}
    </main>
  )
}
