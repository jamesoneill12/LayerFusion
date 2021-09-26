"""computes temporal activation regularization from https://github.com/kondiz/fraternal-dropout/blob/master/main.py"""


def compute_tar(args, data, targets,
                model, hidden, output, loss,
                criterion, ntokens, dropped_rnn_h,
                rnn_h):

    # Kappa penalty
    if args.kappa > 0:
        dm_e = not args.same_mask_e
        dm_i = not args.same_mask_i
        dm_w = not args.same_mask_w
        dm_o = not args.same_mask_o

        if args.eval_auxiliary:
            model.eval()

        kappa_output, _, _, _ = model(data, hidden, return_h=True,
                                      draw_mask_e=dm_e, draw_mask_i=dm_i, draw_mask_w=dm_w, draw_mask_o=dm_o)

        if args.double_target:
            loss = loss + criterion(kappa_output.view(-1, ntokens), targets)
            loss = loss / 2

        l2_kappa = (output - kappa_output).pow(2).mean()
        loss = loss + args.kappa * l2_kappa

    # Activiation Regularization
    l2_alpha = dropped_rnn_h.pow(2).mean()
    loss = loss + args.alpha * l2_alpha

    # Temporal Activation Regularization (slowness)
    loss = loss + args.beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean()

    return loss