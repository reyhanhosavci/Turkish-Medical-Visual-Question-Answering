#bismillah
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"         
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ["PYTHONHASHSEED"] = "42"

import numpy as np
import torch
import torchvision.transforms as T
from torchvision.transforms import functional as F
from PIL import Image
import pandas as pd
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel
import torchvision.models as models
import torch.nn as nn
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
from torch import mul, cat, tanh, relu
from transformers import ViTModel, ViTImageProcessor
from torchvision.transforms.functional import to_pil_image
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix, classification_report
import matplotlib
matplotlib.use("Agg")


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

num_of_unfreeze_block= 3
print("Number of unfreeze blocks: ", num_of_unfreeze_block)

def load_and_process_image(image_path):
    """
    Loads image from path and converts to Tensor, you can also reshape the im
    """
    size = (224, 224)
    img = Image.open(image_path)  # SOR!!
    img = img.resize(size)  
    img = F.to_tensor(img)
    return img


def read_images(paths):
    """
    paths is a dict mapping image ID to image path
    Returns a dict mapping image ID to the processed image
    """
    ims = {}
    for image_id, image_path in paths.items():
        ims[image_path] = load_and_process_image(image_path)
    return ims

def read_txt_file(txt_path, img_path_prefix):
    """
    Expected format: synpicID|question(TR)|answer(TR) 
    """
    rows = []
    with open(txt_path, "r",encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) < 3:
                continue
            syn_id, q_tr, a_tr = parts[0], parts[-2], parts[-1].strip()

            img_path = img_path_prefix+'\\'+syn_id+'.jpg'
            rows.append((img_path, q_tr, a_tr))
    df = pd.DataFrame(rows, columns=["img_path", "question_tr", "answer_tr"])
    return df

def parse_line(line: str):
    if not line or line.strip().startswith("#"):
        return None
    parts = [p.strip() for p in line.rstrip("\n").split("|")]
    # 2 options - (id|question|answer) or (id|category|question|answer)
    if len(parts) < 2:
        return None
    answer = parts[-1].strip()
    if not answer:
        return None
    return answer

def collect_answers(txt_files):
    all_answers = []
    for fpath in txt_files:
        with open(fpath, "r",encoding='utf-8') as f:
            for line in f:
                ans = parse_line(line)
                if ans is not None:
                    all_answers.append(ans)

    uniq = sorted(set(all_answers), key=lambda x: x.casefold())
    return uniq

# Helper for masked mean-pooling over token embeddings
def masked_mean_pool(last_hidden_state, attention_mask):
    # last_hidden_state: [B, T, H], attention_mask: [B, T]
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)  # [B, T, 1]
    summed = (last_hidden_state * mask).sum(dim=1)                  # [B, H]
    denom = mask.sum(dim=1).clamp(min=1e-6)                         # [B, 1]
    return summed / denom # [B, H]

# Freeze all layers, then unfreeze last n blocks only 
def unfreeze_last_n_blocks(hf_model, name: str, n: int = 1):
    # freeze everything
    for p in hf_model.parameters():
        p.requires_grad = False

    # unfreeze last encoder block depending on architecture
    if hasattr(hf_model, "encoder"):
        enc = hf_model.encoder
        # for SBERT/BERT/RoBERTa 
        if hasattr(enc, "layer"):
            for p in enc.layer[-n:].parameters():
                p.requires_grad = True
        # for T5 
        if hasattr(enc, "block"):
            for p in enc.block[-n:].parameters():
                p.requires_grad = True
    
    else:
        # fall back: if we can't detect, keep it frozen (safe default)
        pass


from transformers import AutoTokenizer, AutoModel





def build_text_backbone(encoder_name: str):
    """
    Returns: tokenizer, hf_model, hidden_size
    Only last transformer block is trainable.
    """
    if encoder_name == "BERT":
        model_id = "bert-base-uncased"
        tok = AutoTokenizer.from_pretrained(model_id)
        mdl = AutoModel.from_pretrained(model_id)
        hidden = mdl.config.hidden_size #768

    elif encoder_name == "T5":
        # We'll use encoder output only
        model_id = "t5-base"
        tok = AutoTokenizer.from_pretrained(model_id)
        mdl = AutoModel.from_pretrained(model_id)  # T5Model also works; AutoModel routes to T5ForConditionalGeneration,
        hidden = mdl.config.d_model #768
    
    elif encoder_name == 'XLM-R':  # embed_dim: 768
        model_id = "xlm-roberta-base"
        tok = AutoTokenizer.from_pretrained(model_id)
        mdl = AutoModel.from_pretrained(model_id)
        hidden = mdl.config.hidden_size #768

    elif encoder_name == 'LaBSE':  # embed_dim: 768
        model_id = "sentence-transformers/LaBSE"
        tok = AutoTokenizer.from_pretrained(model_id)
        mdl = AutoModel.from_pretrained(model_id)
        hidden = mdl.config.hidden_size #768

    elif encoder_name == 'BERTurk':  # embed_dim: 768
        model_id = "dbmdz/bert-base-turkish-cased"
        tok = AutoTokenizer.from_pretrained(model_id)
        mdl = AutoModel.from_pretrained(model_id)
        hidden = mdl.config.hidden_size #768
    
    else:
        raise ValueError("Unknown text encoder for backbone build")

    # keep only last block trainable
    unfreeze_last_n_blocks(mdl, encoder_name, num_of_unfreeze_block)
    return tok, mdl, hidden

def check_trainable_layers(mdl):
    if hasattr(mdl, "encoder") and hasattr(mdl.encoder, "layer"):
        total = len(mdl.encoder.layer)
        trainable = [i for i, l in enumerate(mdl.encoder.layer)
                     if any(p.requires_grad for p in l.parameters())]
        print(f"Trainable transformer blocks: {trainable} of {total}")
    elif hasattr(mdl, "transformer") and hasattr(mdl.transformer, "layer"):
        total = len(mdl.transformer.layer)
        trainable = [i for i, l in enumerate(mdl.transformer.layer)
                     if any(p.requires_grad for p in l.parameters())]
        print(f"Trainable transformer blocks: {trainable} of {total}")
    elif hasattr(mdl, "encoder") and hasattr(mdl.encoder, "block"):
        total = len(mdl.encoder.block)
        trainable = [i for i, l in enumerate(mdl.encoder.block)
                     if any(p.requires_grad for p in l.parameters())]
        print(f"Trainable transformer blocks: {trainable} of {total}")
    elif hasattr(mdl, "layers"):
        total = len(mdl.layers)
        trainable = [i for i, l in enumerate(mdl.layers)
                     if any(p.requires_grad for p in l.parameters())]
        print(f"Trainable transformer blocks: {trainable} of {total}")

def check_last_block_trainable(mdl):
    if hasattr(mdl, "encoder") and hasattr(mdl.encoder, "layer"):
        last = mdl.encoder.layer[-1]
    elif hasattr(mdl, "transformer") and hasattr(mdl.transformer, "layer"):
        last = mdl.transformer.layer[-1]
    elif hasattr(mdl, "encoder") and hasattr(mdl.encoder, "block"):
        last = mdl.encoder.block[-1]
    elif hasattr(mdl, "layers"):
        last = mdl.layers[-1]
    else:
        raise RuntimeError("Son blok bulunamadı")
    print("Last block trainable?:", any(p.requires_grad for p in last.parameters()))

# Runs HF encoder -> masked mean-pool -> small head -> 32-d 
class HFTextEncoderHead(nn.Module):
    def __init__(self, hf_model, hidden_size, model_name='SBert'):
        super().__init__()
        self.hf_model = hf_model
        self.model_name = model_name
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.Tanh(),
            nn.Linear(512, 128)
        )

    def forward(self, input_ids, attention_mask):
        if self.model_name == 'T5':
            # T5: use encoder outputs
            outputs = self.hf_model.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            last_hidden = outputs.last_hidden_state  # [B,T,H]
        else:
            outputs = self.hf_model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            last_hidden = outputs.last_hidden_state  # [B,T,H]

        pooled = masked_mean_pool(last_hidden, attention_mask)  # [B,H]
        return self.proj(pooled)  # [B,128]

class ViTImageEncoder(nn.Module):
    def __init__(self, model_name="google/vit-base-patch16-224-in21k"):
        super(ViTImageEncoder, self).__init__()
        self.vit = ViTModel.from_pretrained(model_name)
        num_ftrs = self.vit.config.hidden_size 
        print("For ",model_name,":", num_ftrs) # 768
        self.head =  nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.Tanh(),
            nn.Linear(512, 128)
        )
        # ensure trainable
        for p in self.parameters():
            p.requires_grad = True

    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)
        cls_token = outputs.last_hidden_state[:, 0]
        return self.head(cls_token)

def get_image_encoder(name):
    if name == 'ResNet':
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        num_ftrs = model.fc.in_features
        print("For ",name,":", num_ftrs) # 2048
        model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.Tanh(),
            nn.Linear(512, 128)
        )
        for p in model.parameters():
            p.requires_grad = True
        return model

    elif name == 'DenseNet':
        model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        num_ftrs = model.classifier.in_features
        print("For ",name,":", num_ftrs) # 1024
        model.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.Tanh(),
            nn.Linear(512, 128)
        )
        for p in model.parameters():
            p.requires_grad = True
        return model

    elif name == 'ViT':
        return ViTImageEncoder()

    else:
        raise ValueError("Unknown image encoder")


class CustomDataset(Dataset):
    def __init__(self, img, txt, ans):
      self.img = img
      self.txt = txt
      self.ans = ans

    def __len__(self):
        return len(self.ans)

    def __getitem__(self, idx):
      ans = self.ans[idx]
      img = self.img[idx]
      txt = self.txt[idx]
      return img, txt, ans

class VQA(torch.nn.Module):
    def __init__(self, image_encoder, text_encoder_head, num_answers):
        super(VQA, self).__init__()
        self.image_encoder = image_encoder          # outputs [B,128]
        self.text_encoder_head = text_encoder_head  # outputs [B,128]

        # Text projection: 128 -> 256
        self.text_proj = nn.Sequential(
            nn.Linear(128, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.GELU()
        )

        # Fusion: (128 img + 256 text) -> 128
        self.fusion = nn.Sequential(
            nn.Linear(128 + 256, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.GELU()
        )

        self.classifier = nn.Linear(128, num_answers)

    def forward(self, image, text_tokens):
        xi = self.image_encoder(image)  # [B,128]
        xq = self.text_encoder_head(text_tokens["input_ids"], text_tokens["attention_mask"])  # [B,128]
        xq = self.text_proj(xq)         # [B,256]
        fused = torch.cat((xi, xq), dim=1)   # [B,384]
        z = self.fusion(fused)               # [B,128]
        return self.classifier(z)            # [B,num_answers]


def make_collate_fn(tokenizer, max_len=64):
    def collate(batch):
        imgs, qs, ys = zip(*batch)
        imgs = torch.stack(list(imgs))  # [B, C, H, W] or [B,3,224,224] for ViT path
        ys = torch.stack(list(ys)) if isinstance(ys[0], torch.Tensor) else torch.tensor(ys)
        tok = tokenizer(list(qs), padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        return imgs, tok, ys
    return collate

def plot_confmatrix(y_true, y_pred, class_names=None, out_path=None):
    n = int(max(y_true.max(), y_pred.max())) + 1
    if class_names is None:
        class_names = [str(i) for i in range(n)]

    cm = confusion_matrix(y_true, y_pred, labels=list(range(n)))

    plt.figure(figsize=(5.2, 4.2))  
    ax = sns.heatmap(
        cm, annot=True, fmt='g', cmap='Blues', cbar=True,
        xticklabels=class_names, yticklabels=class_names,
        square=True, linewidths=0.5, linecolor='white'
    )
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    
    plt.xticks(rotation=0, fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=200)
        plt.close()
    else:
        plt.show()


def train_loop(model, optimizer, criterion, train_loader):
    model.train()
    #model.text_encoder_head.train()
    model.to(device)
    total_loss, total = 0, 0

    for image, text, label in train_loader:
        image = image.to(device)
        text_tokens = {k: v.to(device) for k, v in text.items()}
        label = label.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        output = model.forward(image, text_tokens)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        # Record metrics
        total_loss += loss.item()
        total += len(label)

    return total_loss / total


def validate_loop(model, criterion, valid_loader):
    model.eval()
   # model.text_encoder_head.eval()
    model.to(device)
    total_loss, total = 0, 0
    num_correct = 0
    num_samples = 0

    with torch.no_grad():
        for image, text, label in valid_loader:
            # get the inputs; data is a list of [inputs, labels]
            image = image.to(device)
            text_tokens = {k: v.to(device) for k, v in text.items()}
            label = label.to(device)

            # Forward pass
            output = model.forward(image, text_tokens)

            # Calculate how wrong the model is
            loss = criterion(output, label)

            # Record metrics
            total_loss += loss.item()
            total += len(label)

            _, preds = torch.max(output, 1)
            true_labels = label
            num_correct += (preds == true_labels).sum().item()
            num_samples += preds.size(0)

    acc = num_correct / num_samples
    return total_loss / total, acc

# ------------------------------
# 4. Training and Evaluation
# ------------------------------
def train_validate_pipeline(text_encoder_name, image_encoder_name, device):

    result_path = f"models_unfreeze_{num_of_unfreeze_block}block_control/{text_encoder_name}_{image_encoder_name}.pth"
    
    # WandB – Config is a variable that holds and saves hyperparameters and inputs
    batch_size = 32         # input batch size for training (default: 64)
    test_batch_size = 32    # input batch size for testing (default: 1000)
    epochs = 200           # number of epochs to train (default: 10)
    lr = 0.01               # learning rate (default: 0.01)
   

    train_path = r"VQA Med TR\C1C2C3_train.txt"
    val_path = r"VQA Med TR\C1C2C3_val.txt"
    test_path  = r"VQA Med TR\C1C2C3_test.txt"

    txt_files = [train_path, val_path]

    train_df = read_txt_file(train_path, img_path_prefix=r"ImageClef-2019-VQA-Med-Training\Train_images")
    val_df = read_txt_file(val_path, img_path_prefix=r"ImageClef-2019-VQA-Med-Validation\Val_images")
    test_df = read_txt_file(test_path, img_path_prefix=r"ImageClef-2019-VQA-Med-Test\Test_images")
    
    
    train_qs, train_answers, train_image_ids = train_df['question_tr'], train_df['answer_tr'], train_df['img_path']
    val_qs, val_answers, val_image_ids = val_df['question_tr'], val_df['answer_tr'], val_df['img_path']
    test_qs, test_answers, test_image_ids = test_df['question_tr'], test_df['answer_tr'], test_df['img_path']

    train_ims = read_images(train_image_ids)
    val_ims = read_images(val_image_ids)
    test_ims = read_images(test_image_ids)
    
    
    if image_encoder_name == "ViT":
        model_name="google/vit-base-patch16-224-in21k"
        processor = ViTImageProcessor.from_pretrained(model_name)

        # ViT model pixel_values expects shape: [B, 3, 224, 224]
        train_pil_images = [to_pil_image(train_ims[id]) for id in train_image_ids]
        val_pil_images = [to_pil_image(val_ims[id]) for id in val_image_ids]
        test_pil_images = [to_pil_image(test_ims[id]) for id in test_image_ids]

        train_pixel_values = processor(images=train_pil_images, return_tensors="pt")["pixel_values"]
        val_pixel_values = processor(images=val_pil_images, return_tensors="pt")["pixel_values"]
        test_pixel_values = processor(images=test_pil_images, return_tensors="pt")["pixel_values"]

        train_X_ims = train_pixel_values  # shape: [N, 3, 224, 224]
        val_X_ims = val_pixel_values
        test_X_ims = test_pixel_values
    else:
        train_X_ims = torch.stack([train_ims[id] for id in train_image_ids])
        val_X_ims = torch.stack([val_ims[id] for id in val_image_ids])
        test_X_ims = torch.stack([test_ims[id] for id in test_image_ids])


    
    all_answers = collect_answers(txt_files=txt_files)

    print("Number of answer: ", len(all_answers))

    train_Y_idx = np.array([all_answers.index(a) for a in train_answers]) # train_answer_indices
    val_Y_idx = np.array([all_answers.index(a) for a in val_answers])     # val_answer_indices
    test_Y_idx = np.array([all_answers.index(a) for a in test_answers])   # test_answer_indices

    train_Y = torch.tensor(train_Y_idx, dtype=torch.long)
    val_Y   = torch.tensor(val_Y_idx, dtype=torch.long)
    test_Y  = torch.tensor(test_Y_idx, dtype=torch.long)
    # train_Y = torch.zeros(len(train_Y_idx), len(all_answers))
    # train_Y[torch.arange(len(train_Y_idx)), train_Y_idx] = 1
    
    # val_Y = torch.zeros(len(val_Y_idx), len(all_answers))
    # val_Y[torch.arange(len(val_Y_idx)), val_Y_idx] = 1

    # test_Y = torch.zeros(len(test_Y_idx), len(all_answers))
    # test_Y[torch.arange(len(test_Y_idx)), test_Y_idx] = 1
    
    tok, hf_text_model, hidden_size = build_text_backbone(text_encoder_name)
    check_last_block_trainable(hf_text_model)
    check_trainable_layers(hf_text_model)

    train_dataset = CustomDataset(train_X_ims, train_qs, train_Y)
    val_dataset = CustomDataset(val_X_ims, val_qs, val_Y)
    test_dataset = CustomDataset(test_X_ims, test_qs, test_Y)

    collate_fn = make_collate_fn(tok)
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    validloader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn)
    testloader = DataLoader(test_dataset, batch_size=test_batch_size, collate_fn=collate_fn)

    image_encoder = get_image_encoder(image_encoder_name)
    text_head = HFTextEncoderHead(hf_text_model, hidden_size, model_name=text_encoder_name)
    model = VQA(image_encoder, text_head, num_answers=len(all_answers)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    train_losses, valid_losses = [], []
    best_acc = 0.0
    early_stopping_counter = 0
    patience = 10

    for epoch in range(epochs):
        train_loss = train_loop(model, optimizer, criterion, trainloader)
        valid_loss, valid_acc = validate_loop(model, criterion, validloader)
        
        tqdm.write(
            f'epoch #{epoch + 1:3d}\ttrain_loss: {train_loss:.4f}\tvalid_loss: {valid_loss:.4f}\tvalid_acc: {valid_acc:.2%}'
        )
        
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        # Save best model
        if valid_acc > best_acc:
            early_stopping_counter = 0
            best_acc = valid_acc
            if text_encoder_name == 'ELMo':
                cpu_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                torch.save(cpu_state, result_path)
            else:
                torch.save(model.state_dict(), result_path)
            print(f"New best model saved at epoch {epoch+1} with accuracy {valid_acc:.2%}")
        else:
            early_stopping_counter +=1
            if early_stopping_counter >= patience:
                print(f"Early stopped at epoch: {epoch}")
                break
        print({"Epoch": epoch, "Training Loss": train_loss, "Validation Loss": valid_loss})
        

    image_encoder = get_image_encoder(image_encoder_name)
    text_head = HFTextEncoderHead(hf_text_model, hidden_size, model_name=text_encoder_name)
    model = VQA(image_encoder, text_head, num_answers=len(all_answers)).to(device)

    state = torch.load(result_path, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()
    #model.text_encoder_head.eval()

    num_correct = 0
    num_samples = 0
    predictions = []
    answers = []

    with torch.no_grad():
        for image, text, label in testloader:
            image = image.to(device)
            text_tokens = {k: v.to(device) for k, v in text.items()}
            label = label.to(device)
            probs = model(image, text_tokens)

            _, prediction = probs.max(1)
            predictions.append(prediction)      

            #answer = torch.argmax(label, dim=1) 
            answers.append(label)

            num_correct += (prediction == label).sum()
            num_samples += prediction.size(0)
            
    answers = torch.cat(answers).cpu().numpy()
    predictions = torch.cat(predictions).cpu().numpy()        

    test_acc = accuracy_score(answers, predictions)
    p_mac, r_mac, f1_mac, _ = precision_recall_fscore_support(answers, predictions, average='macro', zero_division=0)
    p_w,   r_w,   f1_w,   _ = precision_recall_fscore_support(answers, predictions, average='weighted', zero_division=0)

    # # CONFUSION MATRIX (save as png)
    # plot_confmatrix(
    #     y_true=answers,
    #     y_pred=predictions,
    #     class_names=all_answers,  
    #     out_path=f"results/confmat_{text_encoder_name}_{image_encoder_name}.png"
    # )

    # CLASSIFICATION REPORT (txt)
    try:
        report_str = classification_report(
            answers,
            predictions,
            target_names=all_answers,
            zero_division=0
        )
    except ValueError as e:
        print(f"Warning: {e}")
        print("Regenerating report with matching labels...")

        unique_labels = sorted(set(answers) | set(predictions))
        report_str = classification_report(
            answers,
            predictions,
            labels=unique_labels,
            target_names=[all_answers[i] for i in unique_labels],
            zero_division=0
        )
    with open(f"results_unfreeze_{num_of_unfreeze_block}block_control/classification_report_{text_encoder_name}_{image_encoder_name}.txt", "w", encoding="utf-8") as f:
        f.write(report_str)

    return {
        "accuracy": test_acc,
        "precision_macro": p_mac,
        "recall_macro": r_mac,
        "f1_macro": f1_mac,
        "precision_weighted": p_w,
        "recall_weighted": r_w,
        "f1_weighted": f1_w,
        "answers": answers,
        "predictions": predictions,
        "all_answers": all_answers
    }


# ------------------------------
# 5. Run all combinations
# ------------------------------
import os
import pandas as pd
from sklearn.metrics import classification_report

if __name__ == '__main__':
   
    # text_encoders = ['XLM-R','LaBSE','BERTurk', 'MiniLM-L12-multilingual', 'mDeBERTa-v3','RemBERT']
    # text_encoders = ['T5','BERT','XLM-R','LaBSE','BERTurk']
    # image_encoders = ['DenseNet','ResNet', 'ViT']
    # text_encoders = ['LaBSE','T5','BERT','XLM-R','BERTurk']
    image_encoders = ['DenseNet']
    text_encoders = ['BERT']
   

    os.makedirs(f"results_unfreeze_{num_of_unfreeze_block}block_control", exist_ok=True)
    os.makedirs(f"models_unfreeze_{num_of_unfreeze_block}block_control", exist_ok=True)

    results = []

    for ie in image_encoders:# allahümme ya şafi jfsjdj
        for te in text_encoders:
            print(f'\n\nStarting with: Text = {te}, Image = {ie}')
            result_path = f"models_unfreeze_{num_of_unfreeze_block}block_control/{te}_{ie}.pth"

            try:
                metrics = train_validate_pipeline(te, ie, device)

                print(f"Test Accuracy: {round(metrics['accuracy']*100, 2)}")

                results.append({
                    "Text Encoder": te,
                    "Image Encoder": ie,
                    "Test Accuracy": metrics["accuracy"],
                    "Precision (macro)": metrics["precision_macro"],
                    "Recall (macro)": metrics["recall_macro"],
                    "F1 (macro)": metrics["f1_macro"],
                    "Precision (weighted)": metrics["precision_weighted"],
                    "Recall (weighted)": metrics["recall_weighted"],
                    "F1 (weighted)": metrics["f1_weighted"],
                })

            except ValueError as e:
                print(f"Warning: Error occurred for {te}-{ie} -> {e}")
                # Eğer classification_report kısmında etiket uyuşmazlığı hatası gelirse:
                if "does not match size of target_names" in str(e):
                    print("Attempting safe fallback classification report generation...")
                    # Fallback: sadece kullanılan etiketlerle rapor oluştur
                    try:
                        unique_labels = sorted(set(metrics["accuracy"]) | set(metrics["predictions"]))
                        report_str = classification_report(
                          metrics["accuracy"],
                            metrics["predictions"],
                            labels=unique_labels,
                            target_names=[metrics["all_answers"][i] for i in unique_labels],
                            zero_division=0
                        )
                        # Hatalı olana rağmen en azından sonuçları kaydet
                        with open(f"results_unfreeze_{num_of_unfreeze_block}block_control/classification_report_{te}_{ie}_partial.txt", "w", encoding="utf-8") as f:
                            f.write(report_str)
                    except Exception as inner_e:
                        print(f"Inner error during fallback report: {inner_e}")
                continue  # sonraki kombinasyona geç

    # ---- Final Results ----
    df_results = pd.DataFrame(results)
    print("\nFinal Results:\n", df_results)
    df_results.to_excel(f"results_unfreeze_{num_of_unfreeze_block}block_control/test_results.xlsx", index=False)
    #NOT: labse -densenet üzerine 3 unfreeze block  yanlışıkla tekrar eğitildi :') models_unfreze_3block_without_revision içinde