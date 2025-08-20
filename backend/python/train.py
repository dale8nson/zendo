import argparse
from app.services.training import train
import asyncio

async def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', type=str, default='default')
    parser.add_argument('-t', type=str)
    parser.add_argument('-i', type=str, default='photo')
    parser.add_argument('-s', type=int, default=10)
    parser.add_argument('-lr', type=float, default=1e-4)

    args = parser.parse_args()

    await train(collection=args.c, token=args.t, initializer_token=args.i, max_train_steps=args.s, repeats=args.s, lr=args.lr)


if __name__ == "__main__":
    asyncio.run(main())
