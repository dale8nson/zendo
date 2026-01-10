import { Ribbon } from '@/components/Ribbon'
import { ImageEditor } from '@/components/ImageEditor'
import { Preview } from '@/components/Preview'
import { LeftSideBar } from '@/components/LeftSideBar'
import { RightSideBar } from '@/components/RightSideBar'
import { ResizableHandle, ResizablePanel, ResizablePanelGroup } from '@/components/ui/resizable'
import { ImageControlPanel } from '@/components/ImageControlPanel'

export default function HomePage() {
  return (
    <main className="flex flex-col h-screen w-screen bg-gradient-to-br from-black via-zinc-900 to-neutral-800 text-white">
      <header className="h-12 bg-neutral-950  flex items-center text-sm w-full m-0">
        <Ribbon />
      </header>
      <ResizablePanelGroup direction="horizontal" className="flex overflow-hidden w-full h-full">
        <ResizablePanel defaultSize={15}>
          <LeftSideBar />
        </ResizablePanel>
        <ResizableHandle />
        <ResizablePanel
          defaultSize={35}
          className="relative flex items-center border-r border-neutral-800 h-full w-full"
        >
          <ImageEditor />
          <ImageControlPanel />
        </ResizablePanel>
        <ResizableHandle />
        <ResizablePanel
          defaultSize={35}
          className="relative flex items-center border-r border-neutral-800  bg-no-repeat bg-center h-full w-full"
        >
          <Preview />
        </ResizablePanel>
        <ResizableHandle />
        <ResizablePanel defaultSize={15}>
          <RightSideBar />
        </ResizablePanel>
      </ResizablePanelGroup>
    </main>
  )
}
