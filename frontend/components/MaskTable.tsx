'use client'

import { useEffect, useState } from 'react'
import { useAppSelector, useAppDispatch } from '@/lib/hooks'
import { setMaskData } from '@/lib/features/preview/previewSlice'
import { DataTable } from '@/components/DataTable'
import { ColumnDef } from '@tanstack/react-table'
import { Checkbox } from '@/components/ui/checkbox'

export const MaskTable = () => {
  const maskData = useAppSelector((state) => state.preview.maskData)
  const dispatch = useAppDispatch()

  const [data, setData] = useState<Mask[]>([])

  const columns: ColumnDef<Mask>[] = [
    {
      id: 'select',
      header: ({ table }) => (
        <Checkbox
          checked={
            table.getIsAllPageRowsSelected() ||
            (table.getIsSomePageRowsSelected() && 'indeterminate')
          }
          onCheckedChange={(value) => table.toggleAllPageRowsSelected(!!value)}
          aria-label="Select all"
        />
      ),
      cell: ({ row }) => (
        <Checkbox
          checked={row.getIsSelected()}
          onCheckedChange={(value) => {
            row.toggleSelected(!!value)
            dispatch(
              setMaskData(
                maskData.map((mask, index) =>
                  index === row.index ? { ...mask, active: value } : mask
                )
              )
            )
          }}
          aria-label="Select layer"
        />
      ),
      enableSorting: false,
      enableHiding: false,
    },
    {
      accessorKey: 'label',
      header: 'Label',
    },
    {
      accessorKey: 'include',
      header: ({ table }) => (
        <div className="flex items-center justify-center">
          Include
          {/* <label htmlFor="include">Include</label>
          <Checkbox
            id="include"
            checked={
              table
            }
            onCheckedChange={(value) => table(!!value)}
            aria-label="Select all"
          /> */}
        </div>
      ),
      cell: ({ row }) => (
        <Checkbox
          checked={row.original.include}
          onCheckedChange={(checked) => {
            dispatch(
              setMaskData(
                maskData.map((mask, index) =>
                  index === row.index ? { ...mask, include: checked } : mask
                )
              )
            )
          }}
        />
      ),
      enableSorting: false,
      enableHiding: false,
    },
    {
      accessorKey: 'exclude',
      header: ({ table }) => (
        <div className="flex items-center justify-center">
          Exclude
          {/* <label htmlFor="include">Exclude</label>
          <Checkbox
            id="exclude"
            checked={
              table.getIsAllRowsSelected() || (table.getIsSomePageRowsSelected() && 'indeterminate')
            }
            onCheckedChange={(value) => {
              table.toggleAllPageRowsSelected(!!value)
            }}
            aria-label="Select all"
          /> */}
        </div>
      ),
      cell: ({ row }) => (
        <Checkbox
          checked={row.original.exclude}
          onCheckedChange={(checked) => {
            dispatch(
              setMaskData(
                maskData.map((mask, index) =>
                  index === row.index ? { ...mask, exclude: checked } : mask
                )
              )
            )
          }}
        />
      ),
      enableSorting: false,
      enableHiding: false,
    },
  ]

  useEffect(() => {
    if (!maskData || !maskData.length) return
    console.log(`maskData: ${maskData}`)
    const data = maskData.map((mask, index) => ({
      label: mask.label || `m-${index}`,
      active: mask.active,
      include: mask.include,
      exclude: mask.exclude,
    }))

    setData(data || [])
  }, [maskData])

  return <DataTable columns={columns} data={data} />
}
