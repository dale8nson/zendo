'use client'
import {
  use,
  useEffect,
  useRef,
  useState,
  useCallback,
  PointerEventHandler,
  WheelEventHandler,
} from 'react'
import { useAppSelector, useAppDispatch } from '@/lib/hooks'
import { useQuery } from '@tanstack/react-query'
import { Button } from '@/components/ui/button'
import { ResizableHandle, ResizablePanel, ResizablePanelGroup } from '@/components/ui/resizable'
import {
  ContextMenu,
  ContextMenuContent,
  ContextMenuItem,
  ContextMenuTrigger,
} from '@/components/ui/context-menu'

import { ToolPalette } from './ToolPalette'

import {
  setCurrentHistoryIndex,
  setPreviewCanvasData,
  setPreviewCanvasSize,
  setShouldDrawCanvas,
  appendHistory,
  setPreviewStatus,
  setScaledSelectionBox,
  setMaskData,
  setMaskIndex,
  setMaskBox,
  setSelectedMaskData,
  setSelectedMasks,
  nextMask,
  includeMask,
  excludeMask,
  selectLayer,
  setLayerOpacity,
  setLayerVisible,
  appendLayerHistory,
  newImage,
  setLayerHistoryIndex,
  newEmptyLayer,
  updateLayer,
  previousMask,
} from '@/lib/features/preview/previewSlice'
import { toggleDisabled } from '@/lib/features/control-panel/controlPanelSlice'
import { PromptPanel } from '@/components/PromptPanel'
import { Progress } from './ui/progress'
import { setEditorCanvasStatus } from '@/lib/features/image-editor/imageEditorSlice'

export function Preview() {
  console.log('Preview')

  const canvasRef = useRef(null)
  const caption = useAppSelector((state) => state.imageEditor.caption)
  const iterations = useAppSelector((state) => state.controlPanel.generationIterations)
  const scaleRef = useRef(1)
  const fetchingMask = useRef(false)

  const previewCanvasData = useAppSelector((state) => state.preview.previewCanvasData)
  const maskData = useAppSelector((state) => state.preview.maskData)
  const maskBox = useAppSelector((state) => state.preview.maskBox)
  const maskIndex = useAppSelector((state) => state.preview.maskIndex)
  // const maskVisible = useAppSelector((state) => state.controlPanel.maskVisible)
  const selectedMasks = useAppSelector((state) => state.preview.selectedMasks)
  const selectedMaskData = useAppSelector((state) => state.preview.selectedMaskData)
  const progress = useAppSelector((state) => state.preview.progress)
  const history = useAppSelector((state) => state.preview.history)
  const currentHistoryIndex = useAppSelector((state) => state.preview.currentHistoryIndex)
  const editorCanvasData = useAppSelector((state) => state.imageEditor.editorCanvasData)
  const status = useAppSelector((state) => state.preview.status)
  const layers = useAppSelector((state) => state.layerTable.layers)
  const [mask, setMask] = useState<HTMLImageElement | null>(null)
  const scaledSelectionBox = useAppSelector((state) => state.preview.scaledSelectionBox)
  const layerHistory = useAppSelector((state) => state.preview.layerHistory)
  const disabled = useAppSelector((state) => state.controlPanel.disabled)
  const selectedLayer = useAppSelector((state) => state.preview.selectedLayer)
  const rootBbox = useAppSelector((state) => state.preview.rootBbox)

  const dispatch = useAppDispatch()

  const [currentImage, setCurrentImage] = useState<HTMLImageElement | null>(null)
  const [masks, setMasks] = useState<{ [key: string]: HTMLImageElement }[]>([])
  const contextMenuOpen = useRef(false)
  const [contextMenuState, setContextMenuState] = useState<'open' | 'closed'>('closed')
  const [maskVisible, setMaskVisible] = useState(false)
  const [selectionBoxVisible, setSelectionBoxVisible] = useState(false)
  const [layerImages, setLayerImages] = useState<HTMLImageElement[][]>([])

  const fetchingMasks = useRef(false)
  const pointerDownRef = useRef<boolean>(false)
  const selectionBox = useRef([0, 0, 0, 0])
  const pointerCanvasCoords = useRef<number[]>([0, 0])
  const selectionRange = useRef([0, 0, 0, 0])
  const debounce = useRef(false)

  const drawLayers = async () => {
    if (!canvasRef.current || layerHistory.length === 0) return
    const canv = canvasRef.current as HTMLCanvasElement
    const ctx = canv.getContext('2d')
    if (!ctx) return
    ctx.clearRect(0, 0, canv.width, canv.height)

    const images = layerImages[currentHistoryIndex] || []
    for (let i = 0; i < images.length; i++) {
      const image = images[i]
      if (!image || !image.complete || image.naturalWidth === 0 || image.naturalHeight === 0) {
        // Skip images that haven't finished loading or failed to load
        continue
      }
      const layer = layerHistory[currentHistoryIndex][i]

      const [x1, y1, x2, y2] = layer.history[layer.currentLayerHistoryIndex].bbox as Array<number>

      let [x, y, w, h] = [x1, y1, x2 - x1, y2 - y1]

      if (!layer.visible) continue
      const size = Math.min(canv.width, canv.height)
      const scaleX = size / w
      const scaleY = size / h
      const scale = Math.min(scaleX, scaleY)

      ctx.globalAlpha = layer.opacity
      ;[x, y, w, h] = [x, y, w, h].map((n) => n * scale)

      try {
        ctx.drawImage(image, 0, 0, image.width, image.height, x, y, w, h)
      } catch (e) {
        // Guard against rare race conditions where the image becomes broken
        console.warn('drawImage skipped due to image state', e)
      }
    }
  }

  const drawMasks = () => {
    if (!canvasRef.current) return
    const canv = canvasRef.current as HTMLCanvasElement
    const ctx = canv.getContext('2d')
    if (!ctx) return
    console.log(`masks: `, masks)
    if (masks.length && maskVisible) {
      masks.forEach(async (mask, index) => {
        console.log(`maskData[${index}]: `, maskData[index])
        const [x, y, w, h] = maskData[index].canvas_box
        console.log(`maskData[${index}]: `, maskData[index])
        console.log(`masks[${index}]`, masks[index])
        let image = new Image()
        if (maskData[index].include) {
          image = mask.mask
        }
        if (maskData[index].exclude) {
          image = mask.inverted_mask
        }
        if (maskData[index].active) {
          image = mask.segmentation
        }

        try {
          if (!image.complete || image.naturalWidth === 0) {
            await image.decode().catch(() => {})
          }
          if (image.naturalWidth && image.naturalHeight) {
            ctx.drawImage(image, 0, 0, image.width, image.height, x, y, w, h)
          }
        } catch (e) {
          console.warn('drawMasks skipped image due to load state', e)
        }
      })
    }
  }

  const drawSelectionBox = () => {
    if (!canvasRef.current) return
    const canv = canvasRef.current as HTMLCanvasElement
    const ctx = canv.getContext('2d')
    if (!ctx) return

    let [x, y, w, h] = selectionBox.current

    if (selectionBoxVisible) {
      ctx.strokeStyle = 'white'
      ctx.setLineDash([15, 15])
      ctx.lineWidth = 2
      ctx.strokeRect(x, y, w, h)
      ;[x, y, w, h] = selectionRange.current

      ctx.strokeStyle = 'red'
      ctx.strokeRect(x, y, w, h)
    }
  }

  const pointerDown: PointerEventHandler<HTMLCanvasElement> = (e) => {
    if (!canvasRef.current || fetchingMasks.current || !layerHistory.length) return
    console.log('pointerDown')
    const canv = canvasRef.current as HTMLCanvasElement

    pointerDownRef.current = true
    const { x: bx, y: by, top, left, width, height } = canv.getBoundingClientRect()
    let x = e.clientX - bx
    let y = e.clientY - by
    pointerCanvasCoords.current = [x, y, 0, 0]
    const scaleX = canv.width / width
    const scaleY = canv.height / height

    // x *= scaleX
    // y *= scaleY

    console.log(
      `bx: ${bx}, by: ${by}, e.clientX: ${e.clientX} e.clientY: ${e.clientY}, x: ${x} y: ${y}`
    )

    // selectionBox.current = [x, y, 0, 0]
    setPreviewStatus({
      selection: `x: ${x.toFixed(2)} y: ${y.toFixed(2)} size: 0x0`,
    })
  }

  const pointerMove: PointerEventHandler<HTMLCanvasElement> = (e) => {
    if (
      pointerDownRef.current &&
      !contextMenuOpen.current &&
      canvasRef.current &&
      !fetchingMasks.current &&
      layerHistory.length
    ) {
      const canv = canvasRef.current as HTMLCanvasElement
      let x1,
        y1
        // if (selectionBoxVisible) {
        // ;[x1, y1] = selectionBox.current
        // } else {
      ;[x1, y1] = pointerCanvasCoords.current
      // }

      const { x: bx, y: by, left, top, width, height } = canv.getBoundingClientRect()
      let x2 = e.clientX - bx
      let y2 = e.clientY - by
      const scaleX = canv.width / width
      const scaleY = canv.height / height

      x1 *= scaleX
      y1 *= scaleY
      x2 *= scaleX
      y2 *= scaleY

      let x = Math.min(x1, x2)
      let y = Math.min(y1, y2)
      // let x = x1
      // let y = y1
      let w = Math.abs(x2 - x1)
      let h = Math.abs(y2 - y1)

      // x *= scaleX
      // y *= scaleY
      // w *= scaleX
      // h *= scaleY

      // pointerCanvasCoords.current = [x, y]
      selectionBox.current = [x, y, w, h]
      if (currentImage) {
        selectionRange.current = [
          Math.max(x - w * 0.25, 0),
          Math.max(y - h * 0.25, 0),
          Math.min(w * 1.25, currentImage.width),
          Math.min(h * 1.25, currentImage.height),
        ]
      }
      if (!selectionBoxVisible) setSelectionBoxVisible(true)
      dispatch(
        setPreviewStatus({
          selection: `x: ${x.toFixed(2)} y: ${y.toFixed(2)} size:${w.toFixed(0)}x${h.toFixed(0)}`,
        })
      )
      drawLayers()
      drawMasks()
      drawSelectionBox()
    }
  }

  const pointerUp: PointerEventHandler<HTMLCanvasElement> = (e) => {
    if (
      !canvasRef.current ||
      !pointerDownRef.current ||
      fetchingMasks.current ||
      !layerHistory.length
    )
      return
    console.log('pointerUp')

    const canv = canvasRef.current as HTMLCanvasElement
    pointerDownRef.current = false

    let [x1, y1] = selectionBox.current
    const { x: bx, y: by, left, top, width, height } = canv.getBoundingClientRect()
    let x2 = e.clientX - bx
    let y2 = e.clientY - by
    let x = Math.min(x1, x2)
    let y = Math.min(y1, y2)
    let w = Math.abs(x2 - x1)
    let h = Math.abs(y2 - y1)

    // let scaleX = canv.width / width
    // let scaleY = canv.height / height
    // let scale = Math.min(scaleX, scaleY)

    if (w < 10 && h < 10) {
      if (maskData.length) dispatch(nextMask())
      setSelectionBoxVisible(false)
      // drawLayers()
      // drawMasks()
      // drawSelectionBox()

      return
    }

    drawLayers()
    drawMasks()
    drawSelectionBox()

    // selectionBox.current = [x, y, w, h]
    // dispatch(setMaskBox(selectionBox.current))

    const rootLayer = layerHistory[currentHistoryIndex].find((layer) => layer.label === 'root')
    let index = rootLayer?.currentLayerHistoryIndex as number
    let bbox = rootLayer?.history[index].bbox

    console.log('rootLayer?.history[index].bbox', bbox)
    ;[x1, y1, x2, y2] = bbox as number[]
    ;[x, y, w, h] = [x1, y1, x2 - x1, y2 - y1]

    let scaleX = canv.width / w
    let scaleY = canv.height / h
    let scale = Math.min(scaleX, scaleY)

    dispatch(setMaskBox(selectionBox.current))
    dispatch(setScaledSelectionBox(selectionBox.current.map((n) => n / scale)))

    drawLayers()
    drawMasks()
    drawSelectionBox()
  }

  const maskItemSelectHandler = (e) => {
    console.log('maskItemSelectHandler')
    e.stopPropagation()

    fetchingMasks.current = true

    if (!canvasRef.current) return
    const canv = canvasRef.current as HTMLCanvasElement

    const layer = layerHistory[currentHistoryIndex].find((layer) => layer.selected) as Layer

    const ctx = canv.getContext('2d')
    if (!ctx) return

    drawLayers()
    drawMasks()
    drawSelectionBox()

    const ws = new WebSocket('ws://127.0.0.1:8000/ws/mask')
    const rootLayer = layerHistory[currentHistoryIndex].find((layer) => layer.label === 'root')
    let index = rootLayer?.currentLayerHistoryIndex as number
    let bbox = rootLayer?.history[index].bbox as number[]
    const [x1, y1, x2, y2] = bbox as number[]
    let [x, y, w, h] = [x1, y1, x2 - x1, y2 - y1]
    const size = Math.max(w, h)
    const [mx, my] = [Math.floor((size - w) / 2), Math.floor((size - h) / 2)]
    ;[x, y, w, h] = scaledSelectionBox
    bbox = [x, y, x + w, y + h]
    const message = {
      layer,
      bbox,
    }

    ws.onerror = (event) => {
      console.error('WebSocket error', event)
      ws.close()
      dispatch(toggleDisabled(false))
    }
    ws.onclose = (event) => {
      console.log('WebSocket closed', event)
      dispatch(toggleDisabled(false))
      fetchingMasks.current = false
    }

    ws.onopen = () => {
      ws.send(JSON.stringify(message))

      dispatch(toggleDisabled(true))
      fetchingMask.current = true
      drawSelectionBox()
      ws.onmessage = (event) => {
        const data = JSON.parse(event.data)
        console.log(`data: `, data)
        if (!Object.hasOwn(data, 'status')) {
          console.log(`data: `, data)

          let new_data = [...data]
          let oldData = [...maskData]
          let newMaskData = [...oldData, ...new_data].toSorted((d1, d2) => d2.area - d1.area)
          let [x1, y1, x2, y2] = rootBbox
          let [x, y, w, h] = [x1, y1, x2 - x1, y2 - y1]
          const scaleX = canv.width / w
          const scaleY = canv.height / h
          const scale = Math.min(scaleX, scaleY)
          newMaskData = newMaskData.map((d) => {
            const bbox = d.bbox
            const canvas_bbox = bbox.map((n) => n * scale)
            ;[x1, y1, x2, y2] = canvas_bbox
            // const canvas_box = [x1, y1, x2 - x1, y2 - y1]
            // const [x1, y1, x2, y2] = scaledBbox
            const canvas_box = selectionBox.current
            return { ...d, active: false, canvas_box }
          })
          newMaskData[0].active = true
          dispatch(setMaskData([...newMaskData]))
          dispatch(setMaskIndex(0))

          // if (canvasRef.current) {
          //   drawLayers()
          //   drawMasks()
          //   drawSelectionBox()
          //   const canv = canvasRef.current as HTMLCanvasElement
          //   const ctx = canv.getContext('2d')
          //   if (!ctx) return
          //   const image = new Image()
          //   image.src = `data:image/png;base64,${new_data[0].segmentation}`
          //   image.onload = () => {
          //     const [x, y, w, h] = new_data[0].canvas_box
          //     ctx.drawImage(image, 0, 0, image.width, image.height, x, y, w, h)
          //   }
          setMaskVisible(true)
          setSelectionBoxVisible(false)
          ws.close()
        } else {
          console.log('WebSocket message received:', data)
        }
      }
      // drawLayers()
      // drawMasks()
      // drawSelectionBox()
    }
  }

  const saveFileHandler = async (e) => {}

  const canvasWheelHandler: WheelEventHandler = (e) => {
    if (debounce.current || !masks.length || !maskVisible) return

    e.stopPropagation()

    debounce.current = true

    console.log(`e.deltaY: ${e.deltaY}`)

    if (e.deltaY <= 0) dispatch(nextMask())
    else dispatch(previousMask())

    drawLayers()
    drawMasks()
    drawSelectionBox()

    setTimeout(() => (debounce.current = false), 1000)
  }

  useEffect(() => {
    if (!canvasRef.current) return
    const canv = canvasRef.current as HTMLCanvasElement
    canv.width = canv.clientWidth * devicePixelRatio
    canv.height = canv.clientHeight * devicePixelRatio
    const ctx = canv.getContext('2d')
    if (!ctx) return
    ctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0)
    const { width, height } = canv
    const bbox = [0, 0, width, height]
    console.log(`bbox: `, bbox)
    dispatch(setPreviewCanvasSize(bbox))

    const observer = new ResizeObserver((entries) => {
      if (!canvasRef.current || !maskData.length) return
      const canv = canvasRef.current as HTMLCanvasElement
      const width = entries[0].borderBoxSize[0].inlineSize
      const height = entries[0].borderBoxSize[0].blockSize
      canv.width = width
      canv.height = height
      const bbox = [0, 0, width, height]
      console.log(`bbox: `, bbox)
      dispatch(setPreviewCanvasSize(bbox))
      let [x1, y1, x2, y2] = rootBbox
      let [x, y, w, h] = [x1, y1, x2 - x1, y2 - y1]
      const scaleX = width / w
      const scaleY = height / h
      const scale = Math.min(scaleX, scaleY)
      const newMaskData = maskData.map((data) => {
        const bbox = data.bbox
        const scaledBbox = bbox.map((n) => n * scale)
        ;[x1, y1, x2, y2] = scaledBbox
        const canvas_box = [x1, y1, x2 - x1, y2 - y1]
        return { ...data, canvas_box }
      })

      dispatch(setMaskData(newMaskData))

      // drawLayers()
      // drawMasks()
      // drawSelectionBox()
    })

    // observer.observe(canv)

    return () => observer.unobserve(canv)
  }, [])

  useEffect(() => {
    if (!layerHistory.length) return
    console.log('layerHistory: ', layerHistory)

    const loadImage = (src: string) => new Promise<HTMLImageElement>((resolve, reject) => {
      const img = new Image()
      img.onload = () => resolve(img)
      img.onerror = (e) => reject(e)
      img.src = src
    })

    const run = async () => {
      const layers = layerHistory[currentHistoryIndex]
      const promises = layers.map((layer) =>
        loadImage(`data:image/png;base64,${layer.history[layer.currentLayerHistoryIndex].imageData}`)
          .catch((e) => {
            console.error('Image load failed', e)
            // Return a dummy 1x1 transparent image to keep indexes aligned
            const dummy = new Image()
            dummy.src = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8Xw8AAlEB9lCk6CwAAAAASUVORK5CYII='
            return dummy
          })
      )
      const images = await Promise.all(promises)

      const layerImgs = [...layerImages]
      layerImgs[currentHistoryIndex] = images
      setLayerImages(layerImgs)
    }

    // Kick off async image loading
    run()
  }, [layerHistory])

  useEffect(() => {
    drawLayers()
    drawMasks()
    drawSelectionBox()
  }, [currentHistoryIndex])

  useEffect(() => {
    if (!canvasRef.current || !layerImages) return
    console.log('useEffect layerImages')
    console.log('layerImages: ', layerImages)

    const canv = canvasRef.current as HTMLCanvasElement
    const ctx = canv.getContext('2d')
    if (!ctx) return

    if (maskData.length) {
      let [x1, y1, x2, y2] = rootBbox
      let [x, y, w, h] = [x1, y1, x2 - x1, y2 - y1]
      const scaleX = canv.width / w
      const scaleY = canv.height / h
      const scale = Math.min(scaleX, scaleY)
      const newMaskData = maskData.map((data) => {
        const bbox = data.bbox
        const scaledBbox = bbox.map((n) => n * scale)
        ;[x1, y1, x2, y2] = scaledBbox
        const canvas_box = [x1, y1, x2 - x1, y2 - y1]
        return { ...data, canvas_box }
      })

      dispatch(setMaskData(newMaskData))
    }

    drawLayers()
    drawMasks()
    drawSelectionBox()
    
    const observer = new ResizeObserver((entries, target) => {
      if (maskData.length) {
        const width = entries[0].borderBoxSize[0].inlineSize
        const height = entries[0].borderBoxSize[0].blockSize
        canv.width = width
        canv.height = height
        let [x1, y1, x2, y2] = rootBbox
        let [x, y, w, h] = [x1, y1, x2 - x1, y2 - y1]
        const scaleX = width / w
        const scaleY = height / h
        const scale = Math.min(scaleX, scaleY)
        const newMaskData = maskData.map((data) => {
          const bbox = data.bbox
          const scaledBbox = bbox.map((n) => n * scale)
          ;[x1, y1, x2, y2] = scaledBbox
          const canvas_box = [x1, y1, x2 - x1, y2 - y1]
          return { ...data, canvas_box }
        })

        dispatch(setMaskData(newMaskData))
      }

      drawLayers()
      drawMasks()
      drawSelectionBox()
    })

    observer.observe(canv)

    return () => {
      observer.disconnect()
    }
  }, [layerImages])

  useEffect(() => {
    console.log(`maskData changed, mask index: ${maskIndex}`)
    if (!maskData.length) return
    console.log(`maskData: `, maskData)
    const maskImages: { [key: string]: HTMLImageElement }[] = []

    for (const data of maskData) {
      console.log(`data: ${data}`)
      // let keys: string[] = ['segmentation', 'mask', 'inverted_mask']

      let images: {
        segmentation: HTMLImageElement
        mask: HTMLImageElement
        inverted_mask: HTMLImageElement
      } = { segmentation: new Image(), mask: new Image(), inverted_mask: new Image() }
      // for (const key of Object.values(keys)) {
      let b64 = data.segmentation
      let url = `data:image/png;base64,${b64}`
      const segmentationImage = new Image()
      segmentationImage.src = url
      segmentationImage.onload = () => {
        images.segmentation = segmentationImage
      }
      b64 = data.mask
      url = `data:image/png;base64,${b64}`
      const maskImage = new Image()
      maskImage.src = url
      maskImage.onload = () => {
        images.mask = maskImage
      }
      b64 = data.inverted_mask
      url = `data:image/png;base64,${b64}`
      const invertedMaskImage = new Image()
      invertedMaskImage.src = url
      invertedMaskImage.onload = () => {
        images.inverted_mask = invertedMaskImage
      }

      maskImages.push(images)
    }
    setMasks([...maskImages])
  }, [maskData])

  useEffect(() => {
    console.log(`masks changed, maskIndex: ${maskIndex}`)
    if (!masks.length) return
    console.log(`useEffect masks`)
    console.log(`masks: `, masks)
    drawLayers()
    drawMasks()
    drawSelectionBox()
  }, [masks])

  // useEffect(() => {
  //   if (fetchingMasks) {
  //     drawLayers()
  //     drawMasks()
  //     drawSelectionBox()
  //   }
  // }, [fetchingMasks])

  // useEffect(() => {
  //   drawLayers()
  //   drawMasks()
  //   drawSelectionBox()
  // }, [selectionBoxVisible])

  // const removeSelect = () => {
  //   dispatch(
  //     setSelectedMaskData(
  //       selectedMaskData.filter((data) => data.id !== maskData[maskIndex].id)
  //     )
  //   )
  // }

  return (
    <ResizablePanelGroup
      direction="vertical"
      className="relative flex flex-col items-center justify-start w-full h-full"
    >
      <ResizablePanel className="w-full h-full">
        <div className="flex h-[48px] min-h-[48px] items-start justify-between w-full bg-neutral-900 p-2">
          <div className="flex justify-start items-center gap-x-2 w-full">
            <button
              onClick={() => {
                console.log('Back button clicked')
                dispatch(
                  setCurrentHistoryIndex(
                    currentHistoryIndex > 0 ? currentHistoryIndex - 1 : layerHistory.length - 1
                  )
                )
              }}
              className="cursor-pointer"
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
              onClick={() => {
                console.log('Forward button clicked')
                dispatch(setCurrentHistoryIndex((currentHistoryIndex + 1) % layerHistory.length))
              }}
              className="cursor-pointer"
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
            <p className="text-white text-sm self-justify-center">
              {currentHistoryIndex + 1}/{layerHistory.length}
            </p>
          </div>
          <div className="flex items-center justify-center">
            <label htmlFor="file">
              <svg
                xmlns="http://www.w3.org/2000/svg"
                height="24px"
                viewBox="0 -960 960 960"
                width="24px"
                fill="#e8eaed"
              >
                <path d="M480-320 280-520l56-58 104 104v-326h80v326l104-104 56 58-200 200ZM240-160q-33 0-56.5-23.5T160-240v-120h80v120h480v-120h80v120q0 33-23.5 56.5T720-160H240Z" />
              </svg>
              <input
                id="file"
                type="file"
                onChange={() => {
                  null
                }}
              />
            </label>
          </div>
          {/* <Progress value={progress} className=" top-0, left-0 w-full h-2" /> */}
        </div>
        <div className="flex flex-col relative justify-start w-full">
          <ContextMenu
            onOpenChange={(open) => {
              console.log(`menu open: ${open}`)
              if (open) contextMenuOpen.current = true
              else setTimeout(() => (contextMenuOpen.current = false), 500)
              drawLayers()
              drawMasks()
              drawSelectionBox()
              pointerDownRef.current = false
            }}
          >
            <ContextMenuTrigger>
              <canvas
                ref={canvasRef}
                className="relative w-full h-full cursor-crosshair object-contain"
                width={1024}
                height={1024}
                onPointerDown={pointerDown}
                onPointerMove={pointerMove}
                onPointerUp={pointerUp}
                onWheel={canvasWheelHandler}
              ></canvas>
            </ContextMenuTrigger>
            <ContextMenuContent className="bg-neutral-800">
              <ContextMenuItem className="text-white" onSelect={(e) => maskItemSelectHandler(e)}>
                Mask
              </ContextMenuItem>
              <ContextMenuItem
                disabled={masks.length === 0}
                onSelect={(e) => {
                  e.stopPropagation()
                  setMaskVisible(!maskVisible)
                }}
              >
                {maskVisible ? 'Hide' : 'Show'}
              </ContextMenuItem>
              <ContextMenuItem
                className="text-white"
                disabled={masks.length === 0}
                onSelect={(e) => {
                  e.stopPropagation()
                  dispatch(
                    setMaskData(
                      maskData.map((data, index) => {
                        const new_data = { ...data }
                        if (index == maskIndex) {
                          new_data.include = true
                          new_data.exclude = false
                        }
                        return new_data
                      })
                    )
                  )
                  let newSelectedMasks = [...selectedMasks]
                  const { id, mask: imageData } = maskData[maskIndex]
                  if (maskIndex > selectedMasks.length - 1) {
                    newSelectedMasks.push({ id, imageData })
                  } else {
                    newSelectedMasks = newSelectedMasks.toSpliced(
                      newSelectedMasks.indexOf(
                        newSelectedMasks.find((m) => m.id === id) as SelectedMask
                      ),
                      1,
                      { id, imageData }
                    )
                  }
                  dispatch(setSelectedMasks(newSelectedMasks))
                }}
              >
                Include
              </ContextMenuItem>
              <ContextMenuItem
                disabled={masks.length === 0}
                onSelect={(e) => {
                  e.stopPropagation()
                  dispatch(
                    setMaskData(
                      maskData.map((data, index) => {
                        const new_data = { ...data }
                        if (index == maskIndex) {
                          new_data.include = false
                          new_data.exclude = true
                        }
                        return new_data
                      })
                    )
                  )
                  let newSelectedMasks = [...selectedMasks]
                  const { id, inverted_mask: imageData } = maskData[maskIndex]
                  if (maskIndex > selectedMasks.length - 1) {
                    newSelectedMasks.push({ id, imageData })
                  } else {
                    newSelectedMasks = newSelectedMasks.toSpliced(
                      newSelectedMasks.indexOf(
                        newSelectedMasks.find((m) => m.id === id) as SelectedMask
                      ),
                      1,
                      { id, imageData }
                    )
                  }
                  dispatch(setSelectedMasks(newSelectedMasks))
                }}
              >
                Exclude
              </ContextMenuItem>
              <ContextMenuItem
                disabled={masks.length === 0}
                onSelect={(e) => {
                  e.stopPropagation()
                  dispatch(
                    setMaskData(
                      maskData.map((data, index) => {
                        const new_data = { ...data }
                        if (index == maskIndex) {
                          new_data.active = true
                          new_data.include = false
                          new_data.exclude = false
                        }
                        return new_data
                      })
                    )
                  )
                  let newSelectedMasks = [...selectedMasks]
                  const { id, segmentation: imageData } = maskData[maskIndex]
                  if (maskIndex > selectedMasks.length - 1) {
                    newSelectedMasks.push({ id, imageData })
                  } else {
                    newSelectedMasks = newSelectedMasks.toSpliced(
                      newSelectedMasks.indexOf(
                        newSelectedMasks.find((m) => m.id === id) as SelectedMask
                      ),
                      1,
                      { id, imageData }
                    )
                  }
                  dispatch(setSelectedMasks(newSelectedMasks))
                }}
              >
                Deselect
              </ContextMenuItem>
              <ContextMenuItem
                onSelect={(e) => {
                  e.stopPropagation()
                  dispatch(
                    setMaskData(
                      maskData.map((data, index) => {
                        const new_data = { ...data }
                        new_data.include = false
                        new_data.exclude = false
                        return new_data
                      })
                    )
                  )
                  let newSelectedMasks = [...selectedMasks]
                  const { id, segmentation: imageData } = maskData[maskIndex]
                  if (maskIndex > selectedMasks.length - 1) {
                    newSelectedMasks.push({ id, imageData })
                  } else {
                    newSelectedMasks = newSelectedMasks.toSpliced(
                      newSelectedMasks.indexOf(
                        newSelectedMasks.find((m) => m.id === id) as SelectedMask
                      ),
                      1,
                      { id, imageData }
                    )
                  }
                  dispatch(setSelectedMasks(newSelectedMasks))
                }}
              >
                Deselect All
              </ContextMenuItem>
              <ContextMenuItem
                onSelect={() => {
                  dispatch(setSelectedMasks([]))
                  setMasks([])
                  dispatch(setMaskData([]))
                  drawLayers()
                }}
              >
                Clear
              </ContextMenuItem>
            </ContextMenuContent>
          </ContextMenu>
          <div className="h-[2rem] w-full bg-neutral-800 text-white text-sm p-2 m-0 flex gap-x-4 overflow-clip">
            {Object.entries(status).map((entry) => (
              <span key={entry[0]}>{`${entry[0]}: ${entry[1]}`}</span>
            ))}
          </div>
        </div>
      </ResizablePanel>
      <ResizableHandle className="border-solid border-neutral-800 border-2" />
      <ResizablePanel className="w-full">
        <div className="flex items-center justify-center space-x-4 w-full">
          <ToolPalette />
          {/*<Button
            size="lg"
            aria-label="merge select"
            onClick={() => {
              const selectedMask = { ...maskData[maskIndex] }
              const [x, y, w, h] = scaledSelectionBox
              selectedMask.bbox = [x, y, x + w, y + h]
              selectedMask.canvas_box = maskBox
              dispatch(setSelectedMaskData([...selectedMaskData, selectedMask]))
              dispatch(
                setMaskData([
                  ...maskData.map((data, index) => {
                    const new_data = { ...data }
                    if (index === maskIndex) {
                      new_data.active = false
                      new_data.include = true
                      new_data.exclude = false
                    }
                    return new_data
                  }),
                ])
              )
            }}
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              height="48px"
              viewBox="0 -960 960 960"
              width="48px"
              fill="#e8eaed"
            >
              <path d="M400-80q-33 0-56.5-23.5T320-160v-160H160q-33 0-56.5-23.5T80-400v-400q0-33 23.5-56.5T160-880h400q33 0 56.5 23.5T640-800v160h160q33 0 56.5 23.5T880-560v400q0 33-23.5 56.5T800-80H400Zm0-80h400v-400H560v-240H160v400h240v240Zm80-320Z" />
            </svg>
          </Button>*/}
          {/* <Button size="lg" aria-label="remove select" onClick={removeSelect}>
            <svg
              xmlns="http://www.w3.org/2000/svg"
              height="48px"
              viewBox="0 -960 960 960"
              width="48px"
              fill="#e8eaed"
            >
              <path d="m500-120-56-56 142-142-142-142 56-56 142 142 142-142 56 56-142 142 142 142-56 56-142-142-142 142Zm-220 0v-80h80v80h-80Zm-80-640h-80q0-33 23.5-56.5T200-840v80Zm80 0v-80h80v80h-80Zm160 0v-80h80v80h-80Zm160 0v-80h80v80h-80Zm160 0v-80q33 0 56.5 23.5T840-760h-80ZM200-200v80q-33 0-56.5-23.5T120-200h80Zm-80-80v-80h80v80h-80Zm0-160v-80h80v80h-80Zm0-160v-80h80v80h-80Zm640 0v-80h80v80h-80Z" />
            </svg>
          </Button> */}
        </div>
        <PromptPanel />
      </ResizablePanel>
    </ResizablePanelGroup>
  )
}
