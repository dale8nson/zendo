'use client'

import { useEffect, useState } from 'react'
import { useAppSelector, useAppDispatch } from '@/lib/hooks'
import {
  setMaskData,
  selectLayer,
  setLayerLabel,
  setLayerVisible,
  setLayerOpacity,
} from '@/lib/features/preview/previewSlice'
import { DataTable } from '@/components/DataTable'
import { ColumnDef, createColumnHelper, useReactTable } from '@tanstack/react-table'
import { Checkbox } from '@/components/ui/checkbox'

export const LayerTable = () => {
  const layerHistory = useAppSelector((state) => state.preview.layerHistory)
  const currentHistoryIndex = useAppSelector((state) => state.preview.currentHistoryIndex)
  const dispatch = useAppDispatch()

  const [data, setData] = useState<TableLayerRowData[]>([])
  const [label, setLabel] = useState('')

  const columnHelper = createColumnHelper<TableLayerRowData>()

  const columns = [
    columnHelper.accessor('selected', {
      header: () => 'Active',
      cell: ({ row }) => (
        <input
          type="checkbox"
          checked={row.original.selected}
          onChange={(e) => dispatch(selectLayer(row.index))}
        />
      ),
    }),
    columnHelper.accessor('label', {
      header: () => 'Layer',
      cell: ({ row }) => (
        <input
          type="text"
          value={row.original.label}
          onChange={(e) => setLabel(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && label !== 'root')
              dispatch(setLayerLabel({ index: row.index, label }))
          }}
          disabled={label === 'root'}
        />
      ),
    }),
    columnHelper.accessor('visible', {
      header: () => 'Visible',
      cell: ({ row }) => (
        <input
          type="checkbox"
          checked={row.original.visible}
          onChange={(e) =>
            dispatch(setLayerVisible({ index: row.index, visible: e.target.checked }))
          }
        />
      ),
    }),
    columnHelper.accessor('opacity', {
      header: () => 'Opacity',
      cell: ({ row }) => (
        <input
          type="range"
          min={0}
          max={1}
          step={0.01}
          value={row.original.opacity}
          onChange={(e) => dispatch(setLayerOpacity({ index: row.index, opacity: e.target.value }))}
        />
      ),
    }),
  ]

  useEffect(() => {
    if (!layerHistory.length) return
    let layers = layerHistory[currentHistoryIndex]
    setData(
      layers.map((layer) => {
        const { selected, label, visible, opacity } = layer
        return { selected, label, visible, opacity }
      })
    )
  }, [layerHistory])

  return <DataTable columns={columns} data={data} />
}
