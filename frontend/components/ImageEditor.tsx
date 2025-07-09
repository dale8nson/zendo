'use client'

import { useAppSelector, useAppDispatch } from '@/lib/hooks'
import { MouseEventHandler, useEffect, useRef, useState, WheelEventHandler } from 'react'
import type { MetadataEntry } from './ImageGallery'
import { useQuery, queryOptions, useQueryClient } from '@tanstack/react-query'
import { setSelectedImage, setCaption } from '@/lib/features/image-editor/imageEditorSlice'
import { controller } from '@/lib/utils'
import { setEditorCanvasData } from '@/lib/features/image-editor/imageEditorSlice'
import { Button } from '@/components/Button'
import { ImageControlPanel } from './ImageControlPanel'

interface ScoreRequest {
  filename: string | undefined
  caption: string | null
}

const predict = async (data: MetadataEntry | null) => {
  const response = await fetch('/api/predict', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(data),
  })
  return await response.json()
}

const get_caption = async (data: MetadataEntry) => {
  const response = await fetch('/api/caption', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(data),
  })
  return await response.json()
}

const score = async (data: ScoreRequest | null) => {
  const response = await fetch('/api/score', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(data),
  })
  return await response.json()
}

export const ImageEditor = () => {
  console.log('ImageEditor')
  let selectedImage: MetadataEntry | null = useAppSelector(
    (state) => state.imageEditor.selectedImage
  )
  const [captionScore, setCaptionScore] = useState(null)
  const [captionScoreLoading, setCaptionScoreLoading] = useState(false)

  const canvas = useRef<HTMLCanvasElement>(null)

  const caption = useAppSelector((state) => state.imageEditor.caption)
  const editorCanvasData = useAppSelector((state) => state.imageEditor.editorCanvasData)
  const dispatch = useAppDispatch()
  const queryClient = useQueryClient()

  const textRef = useRef(null)
  const selectionBox = useRef([0, 0, 0, 0])
  const scale = useRef(1)

  const { data } = useQuery(
    queryOptions({
      queryKey: ['caption', selectedImage?.id],
      queryFn: async () => await get_caption(selectedImage as MetadataEntry),
      enabled: !!selectedImage,
      refetchOnWindowFocus: false,
      staleTime: Infinity,
    })
  )

  const { data: caption_score } = useQuery(
    queryOptions({
      queryKey: ['score', selectedImage?.id],
      queryFn: async () => {
        console.log(`filename: ${selectedImage?.filename}, caption: ${data.caption}`)
        const captionScore = await score({
          filename: selectedImage?.filename,
          caption: data.caption,
        })
        return captionScore
      },
      enabled: captionScoreLoading,
      refetchOnWindowFocus: false,
      staleTime: Infinity,
    })
  )

  const drawImage = (
    canv: HTMLCanvasElement,
    image: HTMLImageElement,
    x: number,
    y: number,
    width: number,
    height: number
  ) => {
    console.log(`drawImage`)
    const ctx = canv.getContext('2d')
    ctx?.clearRect(0, 0, width, height)
    console.log(`canvas.width: ${canv.width}, canvas.height: ${canv.height}`)
    let width_scale = canv.width / width
    let height_scale = canv.height / height
    let scale = Math.min(width_scale, height_scale)
    console.log(`scale: ${scale}`)
    // ctx?.scale(scale, scale)
    ctx?.drawImage(
      image,
      x,
      y,
      width,
      height,
      canv.width / 2 - (width * scale) / 2,
      canv.height / 2 - (height * scale) / 2,
      width * scale,
      height * scale
    )
    console.log(`image.src: ${image.src.slice(0, 29)}...`)
    localStorage.setItem('editorCanvasData', image.src)
  }

  const scoreButtonClickHandler: MouseEventHandler<HTMLButtonElement> = async () => {
    if (captionScoreLoading) return
    setCaptionScoreLoading(true)
  }

  const mouseDownHandler = (e: MouseEvent) => {
    const canv = canvas.current as HTMLCanvasElement
    const rect = canv.getBoundingClientRect()
    if (!rect) return
    const x = e.clientX - rect.left
    const y = e.clientY - rect.top
    selectionBox.current = [x, y, 0, 0]

    const mousemove = (e: MouseEvent) => {
      const ctx = canv.getContext('2d')
      if (!ctx) return

      const image = new Image()
      image.src = `data:image/png;base64,${selectedImage?.image_data}`
      image.onload = () => {
        ctx.clearRect(0, 0, canv.width, canv.height)
        drawImage(
          canv,
          image,
          0,
          0,
          selectedImage?.width as number,
          selectedImage?.height as number
        )
        const w = e.clientX - rect.left - x
        const h = e.clientY - rect.top - y
        selectionBox.current = [x, y, w, h]
        ctx.strokeStyle = 'white'
        ctx.setLineDash([15, 15])
        ctx.lineWidth = 2
        ctx.strokeRect(x, y, w, h)
      }
    }
    const mouseUp = () => {
      canv.removeEventListener('mousemove', mousemove)
      canv.removeEventListener('mouseup', mouseUp)
    }
    canv.addEventListener('mouseup', mouseUp)
    canv.addEventListener('mousemove', mousemove)
  }

  useEffect(() => {
    if (!canvas.current) return
    const currentImage = localStorage.getItem('selectedImage')
    if (currentImage) {
      console.log(`currentImage:${currentImage.slice(0, 29)}...`)
      dispatch(setSelectedImage(JSON.parse(currentImage)))
    }
  }, [])

  useEffect(() => {
    if (!selectedImage || !canvas.current) return
    selectedImage = selectedImage as MetadataEntry
    queryClient.invalidateQueries({ queryKey: ['caption', selectedImage?.id] })
    const canv = canvas.current as HTMLCanvasElement
    const { width, height } = canvas.current.getBoundingClientRect()
    canv.width = width
    canv.height = height
    const image = new Image(selectedImage.width, selectedImage.height)
    console.log(`selectedImage.image_data: ${selectedImage.image_data.slice(0, 29)}...`)
    image.src = `data:image/${selectedImage.filename.split('.').pop()};base64,${selectedImage.image_data}`

    image.onload = () => {
      drawImage(canv, image, 0, 0, image.width, image.height)
    }
    console.log(`editorCanvasData: ${selectedImage?.image_data.slice(0, 29)}...`)
    dispatch(setEditorCanvasData(selectedImage?.image_data || ''))
    const observer = new ResizeObserver((entries, target) => {
      canv.width = entries[0].borderBoxSize[0].inlineSize
      canv.height = entries[0].borderBoxSize[0].blockSize
      drawImage(canv, image, 0, 0, image.width, image.height)
      dispatch(setEditorCanvasData(selectedImage?.image_data || ''))
    })
    observer.observe(canv)

    image.onerror = () => {
      console.error('Failed to load image')
    }
  }, [selectedImage])

  useEffect(() => {
    if (data) {
      dispatch(setCaption(data.caption))
    }
  }, [data])

  useEffect(() => {
    if (textRef.current) {
      ;(textRef.current as HTMLTextAreaElement).value = caption || ''
    }
  }, [caption])

  useEffect(() => {
    if (!caption_score) return
    setCaptionScore(caption_score.score || 0)
    setCaptionScoreLoading(false)
  }, [caption_score])

  return (
    <div className="flex flex-col items-start justify-center w-full h-full border-2 border-solid border-neutral-950">
      <div className="flex h-[48px] min-h-[48px] items-start justify-center w-full bg-neutral-900  p-2">
        <h1 className="text-lg font-bold text-white">
          {selectedImage?.original_filename as string}
        </h1>
      </div>
      <div className="flex relative justify-start w-full min-h-[128px] lg:min-h-[256px] xl:min-h-[512px] 2xl:min-h-[1024px]">
        <canvas ref={canvas} className="w-full h-full" onMouseDown={(e) => mouseDownHandler(e)} />
      </div>
      <div className="flex items-start justify-around w-full h-full bg-neutral-900">
        <textarea
          ref={textRef}
          className="text-lg text-white w-4/5 h-full m-0 px-2 resize-none border-2 border-solid border-neutral-800"
          defaultValue={caption || 'Loading...'}
          onChange={(e) => dispatch(setCaption(e.target.value))}
        />
        <div className="flex flex-col items-center justify-between h-full w-1/5 p-2 border-2 border-solid border-neutral-800">
          <h1 className="text-lg font-bold text-white">
            Match Score:{' '}
            {captionScoreLoading
              ? 'Loadin...'
              : captionScore
                ? `${(captionScore as number).toFixed(2)}%`
                : (0).toFixed(2)}
          </h1>
          <Button onClick={scoreButtonClickHandler}>Score</Button>
        </div>
      </div>
    </div>
  )
}
