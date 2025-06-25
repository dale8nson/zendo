'use client'
import { useAppSelector } from '@/lib/hooks'
import { useQuery } from '@tanstack/react-query'
import { useEffect, useRef } from 'react'

const getPreview = async (prompt: string): Promise<string> => {
  const response = await fetch('/api/generate', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Connection: 'keep-alive',
    },
    body: JSON.stringify({ prompt }),
    keepalive: true,
    cache: 'force-cache',
  })
  const data = await response.json()
  return data.image
}

export function Preview() {
  const canvasRef = useRef(null)
  const caption = useAppSelector((state) => state.imageEditor.caption)
  // const { data, isLoading } = useQuery({
  //   queryKey: ['preview', caption],
  //   queryFn: () => getPreview(caption as string),
  //   enabled: !!caption,
  // })

  const drawImage = (
    canv: HTMLCanvasElement,
    image: HTMLImageElement,
    width: number,
    height: number
  ) => {
    console.log(`drawImage`)

    const ctx = canv.getContext('2d')
    ctx?.clearRect(0, 0, width, height)
    console.log(`canvas.width: ${canv.width}, canvas.height: ${canv.height}`)
    const width_scale = canv.width / image.width
    const height_scale = canv.height / image.height
    const scale = Math.min(width_scale, height_scale)
    console.log(`scale: ${scale}`)
    // ctx?.scale(scale, scale)
    ctx?.drawImage(
      image,
      0,
      0,
      image.width,
      image.height,
      canv.width / 2 - (image.width * scale) / 2,
      canv.height / 2 - (image.height * scale) / 2,
      image.width * scale,
      image.height * scale
    )
  }

  const generatePreviewButtonHandler = async () => {
    if (!caption || !canvasRef.current) return
    const canv = canvasRef.current as HTMLCanvasElement
    const ctx = canv.getContext('2d')
    const result = await getPreview(caption as string)
    const img = new Image(512, 512)
    img.src = `data:image/*;base64,${result}`
    img.onload = () => {
      drawImage(canv, img, 1024, 1024)
      const observer = new ResizeObserver((entries, target) => {
        canv.width = entries[0].borderBoxSize[0].inlineSize
        canv.height = entries[0].borderBoxSize[0].blockSize
        drawImage(canv, img, 1024, 1024)
      })
      observer.observe(canv)
    }
  }

  useEffect(() => {
    if (!canvasRef.current) return
    const canv = canvasRef.current as HTMLCanvasElement
    const ctx = canv.getContext('2d')
    const { width, height } = canv.getBoundingClientRect()
  })

  return (
    <div className="flex flex-col justify-center items-center w-full h-full">
      <canvas ref={canvasRef} className="top-0 left-0 w-full h-full border-2 border-dashed" />
      <button
        className="w-full py-2 px-4 rounded-md text-white bg-gradient-to-br from-gray-800 to-black hover:from-black hover:to-gray-900"
        onClick={generatePreviewButtonHandler}
      >
        Generate Preview
      </button>
    </div>
  )
}
