'use client'

import { useState } from 'react'
import {
  Dialog,
  DialogTrigger,
  DialogContent,
  DialogHeader,
  DialogFooter,
  DialogTitle,
  DialogClose,
} from '@/components/ui/dialog'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'

export function FileSaveDialog({ getFileData }: { getFileData: () => string }) {
  const [filename, setFilename] = useState(`image-${crypto.randomUUID()}.png`)

  const handleDownload = () => {
    const blob = new Blob([getFileData()], { type: 'image/png' })
    const url = URL.createObjectURL(blob)

    const a = document.createElement('a')
    a.href = url
    a.download = filename || 'output.png'
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <Dialog>
      <DialogTrigger asChild>
        <Button variant="default">Save File</Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>Save File</DialogTitle>
        </DialogHeader>
        <div className="flex flex-col gap-2 py-4">
          <label className="text-sm font-medium">Filename</label>
          <Input
            value={filename}
            onChange={(e) => setFilename(e.target.value)}
            placeholder="Enter filename (e.g., output.txt)"
          />
        </div>
        <DialogFooter className="sm:justify-end">
          <DialogClose asChild>
            <Button onClick={handleDownload}>Download</Button>
          </DialogClose>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
