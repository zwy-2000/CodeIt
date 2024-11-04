from torch.utils.data import DataLoader
import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm
from torch.nn import CrossEntropyLoss
from codeit.policy.Julian_tokenization.dataloader_custom import DataSetClass


def train_Julian_way(model, train_dataset, tokenizer, n_iter, device):
    model_params = {
        'MODEL': 't5-small',
        'TRAIN_BATCH_SIZE': 4, #reduced from 8
        'VALID_BATCH_SIZE': 4, #reduced from 8
        'TRAIN_EPOCHS': 6,
        'VAL_EPOCHS': 1,
        'LEARNING_RATE': 0.0001,
        'MAX_SOURCE_TEXT_LENGTH': 1024,
        'MAX_TARGET_TEXT_LENGTH': 1024,
        'SEED': 17,
        'NUM_BEAMS': 15,
        'fined_tuned_dir': None
    }


    train_params = {
        'batch_size': model_params["TRAIN_BATCH_SIZE"],
        'shuffle': True,
        'num_workers': 0
    }

    ##    'source_ids' 'source_mask' 'target_ids' 'target_mask' 'name' 'local_path' 'percent_of_seen_pairs'
    ##   the train_dataset might need other preprocessing
    train_dataset = DataSetClass(train_dataset, tokenizer)
    training_loader = DataLoader(train_dataset, **train_params)

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=model_params["LEARNING_RATE"])

    # metrics_blue, metrics_leven, accuracy = train_and_validate(epoch, tokenizer=tokenizer, model=model, device=device,
    #                             loader=training_loader, optimizer=optimizer,
    #                             console=console, cfg=cfg, val_loader_list=val_loader_list)
    
    # let's set not needed var to none for now
    console = None
    val_loader_list = None
    cfg = None

    model, step = train_but_not_validate(epoch = n_iter, tokenizer=tokenizer, model=model, device=device,
                                loader=training_loader, optimizer=optimizer,
                                console=console, cfg=cfg, val_loader_list=val_loader_list)
    
    return model, step


def train_but_not_validate(epoch, tokenizer, model, device, loader, optimizer, console, cfg=None,
      val_loader_list=None):
        
    
    """
    Function to be called for training with the parameters passed from main function
    """
    # output_dir = cfg["output_dir"]
    # test_paths = cfg["test_paths"]
    # train_on_multiple_gpus = cfg["train_on_multiple_gpus"]

    model.train()
    print(f"The model is on the device: {next(model.parameters()).device}")
    percent_of_seen_pairs = 0

    # Initialize the learning rate scheduler
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    iterations = len(loader)


    # ---------------------------- Training Loop ---------------------------- #
    # ----------------------------------------------------------------------- #
    iter_70_percent = int(len(loader) * 0.7)

    # Add tqdm progress bar for the training loop
    progress_bar = tqdm(enumerate(loader, 0), total=len(loader), desc=f"Training Epoch {epoch}", leave=False)
    for step, data in progress_bar:
        # if step > cfg["num_of_itr"] and cfg["num_of_itr"] != -1:
        #     print(f"Number of iterations {step} reached. Stopping the training")
        #     break
        y = data['target_ids'].to(device, dtype=torch.long)
        y_ids = y[:, :-1].contiguous()
        lm_labels = y[:, 1:].clone().detach()
        lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
        ids = data['source_ids'].to(device, dtype=torch.long)
        mask = data['source_mask'].to(device, dtype=torch.long)
        # percent_of_seen_pairs += data["percent_of_seen_pairs"].to(torch.float).sum()


        outputs = model(input_ids=ids, attention_mask=mask, decoder_input_ids=y_ids, labels=lm_labels)
        loss = outputs[0]

        if step > iter_70_percent:
            try:
                loss = weighted_loss(outputs, lm_labels)

            except Exception as e:
                loss = loss
        # if loss.dim() != 0 & train_on_multiple_gpus:
        #     # the loss must be a scaler
        #     loss = loss.mean()

        # if step % 50 == 0:
        # #     wandb.log({"epoch": epoch, "step": step, "loss": loss.item()})
        #     tqdm.write(f"Epoch: {epoch}, Step: {step}, Loss: {loss.item()}")
        progress_bar.set_description(f"Training Epoch {epoch} - Loss: {loss.item():.4f}")




        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(epoch + step /iterations)
    
    return model, step



def weighted_loss(outputs, lm_labels):
    logits = outputs.logits
    loss = 0

    # Flatten the logits and labels for computing cross entropy loss
    batch_size, seq_len, vocab_size = logits.shape
    logits = logits.view(-1, vocab_size)
    lm_labels = lm_labels.view(-1)



    # Create a weight tensor with higher weights for the first tokens
    sequence_length = len(lm_labels)
    weights = torch.linspace(1.5, 0.5, steps=sequence_length).to(lm_labels.device)


    # Calculate the loss using CrossEntropyLoss
    loss_fct = CrossEntropyLoss(ignore_index=-100)
    for i in range(len(lm_labels)):
        if not lm_labels[i] == -100:
            if not torch.isnan(loss_fct(logits[i, :], lm_labels[i])):
                loss += weights[i] * (loss_fct(logits[i, :], lm_labels[i]) / lm_labels[lm_labels != -100].shape[0])
    return loss


