// app/page.tsx

// app/page.tsx

import { ImageUploadForm } from '@/components/ImageUploadForm'
import { ImageGallery } from '@/components/ImageGallery'
import { Logo } from '@/components/Logo'
import { Ribbon } from '@/components/Ribbon'
import { ImageEditor } from '@/components/ImageEditor'

export default function HomePage() {
  return (
    <main className="flex flex-col h-screen w-screen bg-gradient-to-br from-black via-zinc-900 to-neutral-800 text-white">
      <header className="h-12 bg-neutral-950  flex items-center text-sm w-full m-0">
        <Ribbon />
      </header>

      {/* 3-Panel Layout */}
      <div className="flex flex-1 overflow-hidden w-full">
        {/* Left Panel – Gallery */}
        <aside className="w-1/5 min-w-[200px] bg-neutral-900 border-r border-neutral-800 p-2 overflow-y-auto">
          <h2 className="text-xs font-semibold text-neutral-400 mb-2">Upload Image</h2>
          <ImageUploadForm />
          <h2 className="text-xs font-semibold text-neutral-400 mt-6 mb-2">Gallery</h2>
          <ImageGallery />
        </aside>

        {/* Center Panel – Image Editor Placeholder */}
        <section className="relative flex-1 flex-col items-center justify-center border-r border-neutral-800 bg-fill bg-blend-overlay bg-no-repeat bg-center h-full w-3/5 aspect-square">
          <Logo />
          <div className="absolute top-0 left-0 w-full m-auto h-full flex items-center justify-center text-neutral-500 text-lg italic">
            <p className="text-center">TODO: Canvas Editor + Predicted Label Display</p>
          </div>
          <ImageEditor />
        </section>

        {/* Right Panel – Transform / Metadata Placeholder */}
        <aside className="w-1/5 min-w-[220px] max-w-xs bg-neutral-900 p-3">
          <h2 className="text-xs font-semibold text-neutral-400 mb-2">Transform Controls</h2>
          <div className="text-neutral-500 text-sm italic">TODO: Scale / Translate sliders</div>
          <h2 className="text-xs font-semibold text-neutral-400 mt-6 mb-2">Metadata</h2>
          <div className="text-neutral-500 text-sm italic">TODO: Label / Prediction editor</div>
        </aside>
      </div>
    </main>
  )
}

// w-1/5 min-w-[200px] bg-neutral-900 border-r border-neutral-800 p-2 overflow-y-auto
