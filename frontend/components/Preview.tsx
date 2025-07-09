'use client'
import { useAppSelector, useAppDispatch } from '@/lib/hooks'
import { useQuery } from '@tanstack/react-query'
import { use, useEffect, useRef, useState } from 'react'
import { controller } from '@/lib/utils'
import { MetadataEntry } from './ImageGallery'
import { Button } from './Button'
import {
  setCurrentHistoryIndex,
  setPreviewCanvasData,
  setShouldDrawCanvas,
  appendHistory,
} from '@/lib/features/preview/previewSlice'
import { PromptPanel } from '@/components/PromptPanel'
import { Progress } from './ui/progress'

export function Preview() {
  console.log('Preview')

  const canvasRef = useRef(null)
  const caption = useAppSelector((state) => state.imageEditor.caption)
  const i = useAppSelector((state) => state.controlPanel.generationIterations)

  const previewCanvasData = useAppSelector((state) => state.preview.previewCanvasData)
  const maskData = useAppSelector((state) => state.preview.maskData)
  const maskIndex = useAppSelector((state) => state.controlPanel.maskIndex)
  const maskVisible = useAppSelector((state) => state.controlPanel.maskVisible)
  const progress = useAppSelector((state) => state.preview.progress)
  const history = useAppSelector((state) => state.preview.history)
  const currentHistoryIndex = useAppSelector((state) => state.preview.currentHistoryIndex)
  const editorCanvasData = useAppSelector((state) => state.imageEditor.editorCanvasData)

  console.log(
    `editorCanvasData: ${editorCanvasData?.slice(0, 29)} history.length: ${history.length} history[${currentHistoryIndex.value}] ${history[currentHistoryIndex.value]?.slice(0, 29)}`
  )
  const dispatch = useAppDispatch()

  let selectedImage: MetadataEntry | null = useAppSelector(
    (state) => state.imageEditor.selectedImage
  )

  const [drawing, setDrawing] = useState(false)
  const startCoords = useRef([0, 0])
  const endCoords = useRef([0, 0])

  // const drawImage = async (
  //   canv: HTMLCanvasElement,
  //   image: HTMLImageElement,
  //   x: number,
  //   y: number,
  //   width: number,
  //   height: number
  // ) => {
  //   console.log(`drawImage`)
  //   const ctx = canv.getContext('2d')
  //   ctx?.clearRect(0, 0, width, height)
  //   console.log(`canvas.width: ${canv.width}, canvas.height: ${canv.height}`)
  //   let width_scale = canv.width / width
  //   let height_scale = canv.height / height
  //   let scale = Math.min(width_scale, height_scale)
  //   console.log(`scale: ${scale}`)
  //   // ctx?.scale(scale, scale)
  //   ctx?.drawImage(
  //     image,
  //     x,
  //     y,
  //     width,
  //     height,
  //     canv.width / 2 - (width * scale) / 2,
  //     canv.height / 2 - (height * scale) / 2,
  //     width * scale,
  //     height * scale
  //   )

  //   localStorage.setItem('previewCanvasData', image.src)
  // }
  //
  const drawImage = (canv: HTMLCanvasElement, image: HTMLImageElement) => {
    console.log(`drawImage`)
    console.log(`width: ${image.width}, height: ${image.height}`)
    const ctx = canv.getContext('2d')
    const { width: canvasWidth, height: canvasHeight } = canv.getBoundingClientRect()
    console.log(`canvasWidth: ${canvasWidth}, canvasHeight: ${canvasHeight}`)
    canv.width = canvasWidth
    canv.height = canvasHeight

    // ctx?.clearRect(0, 0, canvasWidth, canvasHeight)
    console.log(`canvas.width: ${canv.width}, canvas.height: ${canv.height}`)
    let width_scale = canv.width / image.width
    let height_scale = canv.height / image.height
    let scale = Math.min(width_scale, height_scale)
    console.log(`scale: ${scale}`)
    console.log(`canvasWidth / 2: ${canvasWidth / 2}`)
    console.log(`canvasHeight / 2: ${canvasHeight / 2}`)
    console.log(
      `width / 2: ${image.width / 2} height / 2: ${image.height / 2} (width * scale) / 2: ${(image.width * scale) / 2} (height * scale) / 2: ${(image.height * scale) / 2}`
    )
    ctx?.drawImage(
      image,
      0,
      0,
      image.width,
      image.height,
      canvasWidth / 2 - (image.width * scale) / 2,
      canvasHeight / 2 - (image.height * scale) / 2,
      image.width * scale,
      image.height * scale
    )
    // ctx?.scale(scale, scale)
    console.log(`image.src: ${image.src.slice(0, 29)}...`)
    localStorage.setItem('previewCanvasData', image.src)
  }

  const init = () => {
    console.log(
      `editorCanvasData: ${editorCanvasData?.slice(0, 29)} history.length: ${history.length} history[${currentHistoryIndex.value}] ${history[currentHistoryIndex.value]?.slice(0, 29)}`
    )
    if (!canvasRef.current) return
    const currentImage = localStorage.getItem('previewCanvasData')
    if (currentImage) {
      console.log(`currentImage:${currentImage.slice(0, 29)}...`)
      const canv = canvasRef.current as HTMLCanvasElement
      const ctx = canv.getContext('2d')
      const image = new Image()
      image.src = currentImage
      dispatch(appendHistory(currentImage.split(',', 1)[1]))
      dispatch(setCurrentHistoryIndex(0))
      const scale_width = canv.width / image.width
      const scale_height = canv.height / image.height
      const scale = Math.min(scale_width, scale_height)
      image.onload = () => {
        ctx?.drawImage(image, 0, 0, image.width, image.height, 0, 0, image.width, image.height)
      }
    }
  }

  useEffect(() => {
    if (!canvasRef.current || history.length === 0 || currentHistoryIndex.value >= history.length)
      return
    console.log('Preview updating...')
    if (currentHistoryIndex.value < 0) {
      init()
      return
    }
    const canv = canvasRef.current as HTMLCanvasElement
    const { width, height } = canv.getBoundingClientRect()
    canv.width = width
    canv.height = height
    const ctx = canv.getContext('2d')
    const image = new Image()
    console.log(
      `history[${currentHistoryIndex.value}]: ${history[currentHistoryIndex.value].slice(0, 29)}`
    )
    image.src = `data:image/png;base64,${history[currentHistoryIndex.value]}`

    let width_scale = canv.width / image.width
    let height_scale = canv.height / image.height
    let scale = Math.min(width_scale, height_scale)
    console.log(`scale: ${scale}`)

    image.onload = () => {
      ctx.clearRect(0, 0, canv.width, canv.height)
      drawImage(canv, image, 0, 0, image.width, image.height)
      // drawImage(canv, image, 0, 0, image.width, image.height)
      // localStorage.setItem('previewCanvasData', image.src)
      //
      if (maskVisible && maskData?.[maskIndex]) {
        drawMask()
      }
    }

    const observer = new ResizeObserver((entries, target) => {
      canv.width = entries[0].borderBoxSize[0].inlineSize
      canv.height = entries[0].borderBoxSize[0].blockSize
      width_scale = canv.width / image.width
      height_scale = canv.height / image.height
      scale = Math.min(width_scale, height_scale)
      console.log(`scale: ${scale}`)
      ctx.clearRect(0, 0, canv.width, canv.height)
      drawImage(canv, image)
      if (maskVisible && maskData?.[maskIndex]) {
        drawMask()
      }
    })
    observer.observe(canv)

    function drawMask() {
      const mask = new Image()
      console.log(
        `maskData[${maskIndex}].segmentation: ${maskData?.[maskIndex].segmentation.slice(0, 29)}`
      )

      mask.src = `data:image/png;base64,${maskData?.[maskIndex].segmentation}`
      mask.onload = () => {
        console.log(`mask.width: ${mask.width} mask.height: ${mask.height}`)
        const width_scale = canv.width / mask.width
        const height_scale = canv.height / mask.height
        const scale = Math.min(width_scale, height_scale)
        console.log(`mask scale: ${scale}`)
        const { width, height } = canv.getBoundingClientRect()
        ctx.drawImage(
          mask,
          0,
          0,
          mask.width,
          mask.height,
          width / 2 - (mask.width * scale) / 2,
          height / 2 - (mask.height * scale) / 2,
          mask.width * scale,
          mask.height * scale
        )
      }
    }
  }, [maskData, maskVisible, maskIndex, currentHistoryIndex, history])

  const pointerEnterHandler = (e) => {
    if (!canvasRef.current) return
    const canv = canvasRef.current as HTMLCanvasElement
    const [x, y] = [e.mouseX, e.mouseY]
  }

  const pointerLeaveHandler = () => {
    if (!canvasRef.current) return
    const canv = canvasRef.current as HTMLCanvasElement
  }

  const pointerMoveHandler = () => {
    if (!canvasRef.current) return
    const canv = canvasRef.current as HTMLCanvasElement
  }

  const pointerDownHandler = () => {
    if (!canvasRef.current) return
    const canv = canvasRef.current as HTMLCanvasElement
    const ctx = canv.getContext('2d')
    const { width, height } = canv.getBoundingClientRect()
  }

  return (
    <div className="relative flex flex-col items-center justify-start w-full h-full">
      <div className="flex h-[48px] min-h-[48px] items-start justify-between w-full bg-neutral-900  p-2">
        <div className="flex justify-start items-center gap-x-2">
          <button
            onClick={() => dispatch(setCurrentHistoryIndex(currentHistoryIndex.value - 1))}
            disabled={currentHistoryIndex.value === 0}
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              height="24px"
              viewBox="0 -960 960 960"
              width="24px"
              fill="#e8eaed"
            >
              <path d="m313-440 224 224-57 56-320-320 320-320 57 56-224 224h487v80H313Z" />
            </svg>
          </button>
          <button
            onClick={() => dispatch(setCurrentHistoryIndex(currentHistoryIndex.value + 1))}
            disabled={currentHistoryIndex.value === history.length - 1}
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              height="24px"
              viewBox="0 -960 960 960"
              width="24px"
              fill="#e8eaed"
            >
              <path d="M647-440H160v-80h487L423-744l57-56 320 320-320 320-57-56 224-224Z" />
            </svg>
          </button>
        </div>
        {/* <Progress value={progress} className=" top-0, left-0 w-full h-2" /> */}
      </div>
      <div className="flex flex-col justify-center items-center w-full min-h-[128px] lg:min-h-[256px] xl:min-h-[512px] 2xl:min-h-[1024px] relative">
        <canvas ref={canvasRef} className="relative w-full h-full " />
      </div>
      <PromptPanel />
    </div>
  )
}
