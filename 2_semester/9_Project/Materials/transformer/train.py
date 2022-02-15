import argparse
import datetime

import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn as nn
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from tensorboardX import SummaryWriter
from torch.optim import Adam

from data.dataloader import Dataloader
from other import ScheduledOptim
from transformer import Transformer


def main(gpu, ngpus_per_node, args):
    name = datetime.datetime.now().strftime("_%I:%M%p_%B_%d")

    args.gpu = gpu
    args.rank = args.rank * ngpus_per_node + gpu
    args.batch = int(args.batch / ngpus_per_node)
    args.workers = int(args.workers / ngpus_per_node)
    device = torch.device(args.gpu)

    dist.init_process_group(
        backend="nccl",
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )

    if args.gpu is not None:
        print(
            "Use GPU: {} for training at node with rank {}".format(args.gpu, args.rank)
        )

    if args.rank == 0:
        writer = SummaryWriter("tbs/" + name)

    train_loader = Dataloader("./data/files/ru_train_id", "./data/files/en_train_id")
    if args.rank == 0:
        valid_loader = Dataloader(
            "./data/files/ru_valid_id", "./data/files/en_valid_id"
        )

    if args.rank == 0:
        sp = {"ru": spm.SentencePieceProcessor(), "en": spm.SentencePieceProcessor()}
        for k in sp.keys():
            sp[k].Load("./data/files/{}.model".format(k))

    model = Transformer(
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        clipping_distance=args.clipping_distance,
        dropout=args.dropout,
        encoder_embeddings_path="./data/files/en.npy",
        decoder_embeddings_path="./data/files/ru.npy",
    ).to(device)

    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

    optimizer = Adam(
        [p for p in model.parameters() if p.requires_grad], betas=(0.9, 0.999), eps=1e-6
    )
    scheduler = ScheduledOptim(512, 6000)

    model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

    model.train()
    if args.rank == 0:
        chencherry = SmoothingFunction()

    for i in range(args.num_iterations):

        optimizer.zero_grad()
        scheduler.update_learning_rate(optimizer)

        input, decoder_input, target = train_loader.next_batch(args.batch, device)

        out = model(
            input, decoder_input, checkpoint_gradients=args.checkpoint_gradients
        )
        loss = criterion(out.flatten(0, 1), target.flatten())
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if i % 20 == 0 and args.rank == 0:
            model.eval()

            with torch.no_grad():
                input, decoder_input, target = valid_loader.next_batch(
                    args.batch * 6, device
                )

                out = model(input, decoder_input)
                valid_loss = criterion(out.flatten(0, 1), target.flatten())

                writer.add_scalar("loss", valid_loss, i)
                print("i {}, valid {}".format(i, valid_loss.item()))
                print("_________")

            model.train()

        if i % 160 == 0 and args.rank == 0:
            model.eval()

            input, _, target = valid_loader.next_batch(1, device)

            hyp, ref = [], []

            generations = model(input)
            print("-------source-------")
            print(sp["en"].DecodeIds([int(idx) for idx in input.cpu().numpy()[0]]))
            print("-------target-------")
            print(sp["ru"].DecodeIds([int(idx) for idx in target.cpu().numpy()[0]]))
            print("-----generation-----")
            for generation in generations:
                print(sp["ru"].DecodeIds(generation))

            ref.append(
                [
                    word_tokenize(
                        sp["ru"]
                        .DecodeIds([int(idx) for idx in target.cpu().numpy()[0]])
                        .lower()
                    )
                ]
            )
            hyp.append(word_tokenize(sp["ru"].DecodeIds(generations[0]).lower()))

            try:
                valid_bleu = corpus_bleu(
                    ref, hyp, weights=(0.5, 0.5), smoothing_function=chencherry.method1
                )
                print(f"{i}: bleu: {valid_bleu}")
                writer.add_scalar("bleu", valid_bleu, i)
            except:
                pass

            model.train()

        if i % 2000 == 0 and args.rank == 0:
            torch.save(model.module.state_dict(), "checkpoint/{}.pt".format(name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="bert")
    parser.add_argument("--num-iterations", type=int, default=1000000)
    parser.add_argument("--batch", default=80, type=int)
    parser.add_argument(
        "--n-layers", default=6, type=int, help="number of layers in Transformer"
    )
    parser.add_argument(
        "--n-heads",
        default=8,
        type=int,
        help="number of heads in each layer of Transformer",
    )
    parser.add_argument(
        "--clipping-distance",
        type=int,
        default=8,
        help="maximum relative distance between tokens",
    )

    parser.add_argument("--dropout", type=float, default=0.35, help="dropout rate")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="device where training is going to be performed",
    )
    parser.add_argument(
        "--opt-level", type=str, default="O0", help="opt-level of apex.amp"
    )
    parser.add_argument(
        "--checkpoint-gradients", dest="checkpoint_gradients", action="store_true"
    )
    parser.add_argument(
        "--no-checkpoint-gradients", dest="checkpoint_gradients", action="store_false"
    )
    parser.set_defaults(checkpoint_gradients=True)

    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--world-size", default=1, type=int)
    parser.add_argument("--rank", default=0, type=int)
    parser.add_argument("--dist-url", default="tcp://127.0.0.1:4433", type=str)

    args = parser.parse_args()

    ngpus_per_node = torch.cuda.device_count()

    args.world_size = ngpus_per_node * args.world_size

    torch.multiprocessing.spawn(
        main, nprocs=ngpus_per_node, args=(ngpus_per_node, args)
    )
