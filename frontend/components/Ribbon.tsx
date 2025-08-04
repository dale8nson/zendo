'use client'

import { useState } from 'react'
import { useAppDispatch } from '@/lib/hooks'
import { setCollection } from '@/lib/features/image-editor/imageEditorSlice'

import {
  Menubar,
  MenubarCheckboxItem,
  MenubarContent,
  MenubarItem,
  MenubarMenu,
  MenubarRadioGroup,
  MenubarRadioItem,
  MenubarSeparator,
  MenubarShortcut,
  MenubarSub,
  MenubarSubContent,
  MenubarSubTrigger,
  MenubarTrigger,
} from '@/components/ui/menubar'

import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
  DialogFooter,
  DialogClose,
} from '@/components/ui/dialog'

import { Input } from './ui/input'
import { Button } from './ui/button'
import { Label } from './ui/label'

export function Ribbon() {
  const dispatch = useAppDispatch()

  const [collectiondialogOpen, setCollectionDialogOpen] = useState(false)
  const [collectionField, setCollectionField] = useState('default')

  return (
    <>
      <Menubar className="w-full mx-0 bg-neutral-900 border-t-0 border-x-0 border-b-neutral-800 border-b-2 m-0 p-4 h-full">
        <MenubarMenu>
          <MenubarTrigger>File</MenubarTrigger>
          <MenubarContent className="bg-neutral-900">
            <MenubarItem>
              Open <MenubarShortcut>⌘O</MenubarShortcut>
            </MenubarItem>
            <MenubarItem>
              New Project <MenubarShortcut>⌘N</MenubarShortcut>
            </MenubarItem>
            <MenubarItem>
              New Model <MenubarShortcut>⌘P</MenubarShortcut>
            </MenubarItem>
            <MenubarItem>
              Save <MenubarShortcut>⌘S</MenubarShortcut>
            </MenubarItem>
          </MenubarContent>
        </MenubarMenu>
        <MenubarMenu>
          <MenubarTrigger>Edit</MenubarTrigger>
          <MenubarContent className="bg-neutral-900">
            <MenubarItem>
              Undo <MenubarShortcut>⌘Z</MenubarShortcut>
            </MenubarItem>
            <MenubarItem>
              Redo <MenubarShortcut>⇧⌘Z</MenubarShortcut>
            </MenubarItem>
            <MenubarItem>
              Cut <MenubarShortcut>⌘X</MenubarShortcut>
            </MenubarItem>
            <MenubarItem>
              Copy <MenubarShortcut>⌘C</MenubarShortcut>
            </MenubarItem>
            <MenubarItem>
              Paste<MenubarShortcut>⌘V</MenubarShortcut>
            </MenubarItem>
          </MenubarContent>
        </MenubarMenu>
        <MenubarMenu>
          <MenubarTrigger>View</MenubarTrigger>
          <MenubarContent className="bg-neutral-900">
            <MenubarCheckboxItem>Collection</MenubarCheckboxItem>
          </MenubarContent>
        </MenubarMenu>
        <MenubarMenu>
          <MenubarTrigger>Model</MenubarTrigger>
          <MenubarContent className="bg-neutral-900"></MenubarContent>
        </MenubarMenu>
        <MenubarMenu>
          <MenubarTrigger>Help</MenubarTrigger>
          <MenubarContent className="bg-neutral-900">
            <MenubarItem>About</MenubarItem>
          </MenubarContent>
        </MenubarMenu>
      </Menubar>
      <Dialog open={collectiondialogOpen}>
        <DialogTrigger asChild>
          <Button variant="outline">Share</Button>
        </DialogTrigger>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle>Collection</DialogTitle>
            <DialogDescription>Enter Collection Name to Load </DialogDescription>
          </DialogHeader>
          <div className="flex items-center gap-2">
            <div className="grid flex-1 gap-2">
              <Label htmlFor="collection" className="sr-only">
                Collection
              </Label>
              <Input
                id="collection"
                defaultValue="default"
                value={collectionField}
                onChange={(e) => setCollectionField(e.target.value)}
              />
            </div>
          </div>
          <DialogFooter className="sm:justify-start">
            <DialogClose asChild>
              <Button
                type="button"
                variant="secondary"
                onClick={() => {
                  setCollectionDialogOpen(false)
                  dispatch(setCollection(collectionField))
                }}
              >
                Load
              </Button>
            </DialogClose>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  )
}
