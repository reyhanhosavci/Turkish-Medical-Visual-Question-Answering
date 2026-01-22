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


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"         
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    
class ViTImageEncoder(nn.Module):
    def __init__(self, model_name="google/vit-base-patch16-224-in21k"):
        super(ViTImageEncoder, self).__init__()
        self.vit = ViTModel.from_pretrained(model_name)
        num_ftrs = self.vit.config.hidden_size # 768
        print("For ",model_name,":", num_ftrs)
        self.head =  nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.Tanh(),
            nn.Linear(512, 128)
        )

    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)
        cls_token = outputs.last_hidden_state[:, 0]  # [B, hidden_dim]
        return self.head(cls_token)  # [B, 32]
    
# ------------------------------
# 2. Encoders
# ------------------------------
def get_text_embeddings(encoder_name, texts):
    if isinstance(texts, pd.Series):
        texts = texts.tolist()
    elif isinstance(texts, np.ndarray):
        texts = texts.tolist()
    from transformers import AutoTokenizer, AutoModel
    if encoder_name == 'BERT': # embed_dim: 768
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        model = AutoModel.from_pretrained("bert-base-uncased")
        tokens = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**tokens)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        return cls_embeddings

    elif encoder_name == 'T5': # embed_dim: 768
        tokenizer = AutoTokenizer.from_pretrained("t5-base")
        model = AutoModel.from_pretrained("t5-base")
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model.encoder(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)  # mean of token embedding
        return embeddings
    
    elif encoder_name == 'XLM-R':  # embed_dim: 768
        tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
        model = AutoModel.from_pretrained("xlm-roberta-base")
        tokens = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**tokens)
            last_hidden = outputs.last_hidden_state  # [batch, seq_len, hidden_size]
            input_mask_expanded = tokens['attention_mask'].unsqueeze(-1).expand(last_hidden.size())
            sum_embeddings = torch.sum(last_hidden * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)
            embeddings = sum_embeddings / sum_mask
            return embeddings 
    elif encoder_name == 'XLM-R-Large':  # embed_dim: 1024
        from transformers import AutoTokenizer, AutoModel

        tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")
        model = AutoModel.from_pretrained("xlm-roberta-large")

        # Metinleri tokenize et
        tokens = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

        # GPU kullanımı için:
        tokens = {k: v.to(device) for k, v in tokens.items()}
        model = model.to(device)
        model.eval()

        with torch.no_grad():
            outputs = model(**tokens)
            last_hidden = outputs.last_hidden_state  # [batch, seq_len, 1024]

            # attention mask'ini genişlet
            input_mask_expanded = tokens['attention_mask'].unsqueeze(-1).expand(last_hidden.size())

            # masked mean pooling
            sum_embeddings = torch.sum(last_hidden * input_mask_expanded, dim=1)
            sum_mask = input_mask_expanded.sum(dim=1)
            embeddings = sum_embeddings / sum_mask.clamp(min=1e-9)  # zero division koruması

        return embeddings
    
            
    elif encoder_name == 'LaBSE':  # embed_dim: 768
        from transformers import AutoTokenizer, AutoModel
        tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/LaBSE")
        model = AutoModel.from_pretrained("sentence-transformers/LaBSE")
        tokens = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**tokens)
            last_hidden = outputs.last_hidden_state
            input_mask_expanded = tokens['attention_mask'].unsqueeze(-1).expand(last_hidden.size())
            sum_embeddings = torch.sum(last_hidden * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)
            embeddings = sum_embeddings / sum_mask
        return embeddings        
    
    elif encoder_name == 'BERTurk':  # embed_dim: 768
        tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")
        model = AutoModel.from_pretrained("dbmdz/bert-base-turkish-cased")
        tokens = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**tokens)
            last_hidden = outputs.last_hidden_state
            input_mask_expanded = tokens['attention_mask'].unsqueeze(-1).expand(last_hidden.size())
            sum_embeddings = torch.sum(last_hidden * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)
            embeddings = sum_embeddings / sum_mask
        return embeddings
    
    
    elif encoder_name == 'mDeBERTa-v3':  # embed_dim: 768
        tokenizer = AutoTokenizer.from_pretrained("microsoft/mdeberta-v3-base")
        model = AutoModel.from_pretrained("microsoft/mdeberta-v3-base")
        tokens = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**tokens)
            last_hidden = outputs.last_hidden_state
            input_mask_expanded = tokens['attention_mask'].unsqueeze(-1).expand(last_hidden.size())
            sum_embeddings = torch.sum(last_hidden * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)
            embeddings = sum_embeddings / sum_mask
        return embeddings

    elif encoder_name == 'RemBERT':  # embed_dim: 768
        tokenizer = AutoTokenizer.from_pretrained("google/rembert")
        model = AutoModel.from_pretrained("google/rembert")
        tokens = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**tokens)
            last_hidden = outputs.last_hidden_state
            input_mask_expanded = tokens['attention_mask'].unsqueeze(-1).expand(last_hidden.size())
            sum_embeddings = torch.sum(last_hidden * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)
            embeddings = sum_embeddings / sum_mask
        return embeddings
    
    
    else:
        raise ValueError("Unknown text encoder")


def get_image_encoder(name):
    if name == 'ResNet':
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        num_ftrs = model.fc.in_features # 2048
        print("For ",name,":", num_ftrs)
        model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.Tanh(),
            nn.Linear(512, 128)
        )
        return model

    elif name == 'DenseNet':
        model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        num_ftrs = model.classifier.in_features  # 1024
        print("For ",name,":", num_ftrs)
        model.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.Tanh(),
            nn.Linear(512, 128)
        )
        return model
    
    elif name == 'ViT':
        return ViTImageEncoder()

    else:
        raise ValueError("Unknown image encoder")
    
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

class VQA(torch.nn.Module):
    def __init__(self, image_encoder, embedding_size, num_answers):
        super(VQA, self).__init__()
            
        self.image_encoder = image_encoder
            
            # text projection
        self.text_proj = nn.Sequential(
            nn.Linear(embedding_size, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.GELU(),
        )

            # fusion
        self.fusion = nn.Sequential(
            nn.Linear(128 + 256, 256),  # concat size
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.GELU()
        )
        self.classifier = nn.Linear(128, num_answers)

    def forward(self, x, q):
        x = self.image_encoder(x)               # [B,128]
        q = self.text_proj(q)                   # [B,256]
        z = torch.cat((x, q), dim=1)            # [B,384]
        z = self.fusion(z)                      # [B,128]
        return self.classifier(z)               # [B,num_answers]
  

def train_loop(model, optimizer, criterion, train_loader):
    model.train()
    model.to(device)
    total_loss, total = 0, 0

    for image, text, label in train_loader:
        # get the inputs; data is a list of [inputs, labels]
        image, text, label =  image.to(device), text.to(device), label.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        output = model.forward(image, text)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        # Record metrics
        total_loss += loss.item()
        total += len(label)

    return total_loss / total


def validate_loop(model, criterion, valid_loader):
    model.eval()
    model.to(device)
    total_loss, total = 0, 0
    num_correct = 0
    num_samples = 0

    with torch.no_grad():
      for image, text, label in valid_loader:
          # get the inputs; data is a list of [inputs, labels]
          image, text, label =  image.to(device), text.to(device), label.to(device)

          # Forward pass
          output = model.forward(image, text)

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
    result_path = f"models/{text_encoder_name}_{image_encoder_name}.pth"
    
    # WandB – Config is a variable that holds and saves hyperparameters and inputs
    batch_size = 32         # input batch size for training (default: 64)
    test_batch_size = 32    # input batch size for testing (default: 1000)
    epochs = 200           # number of epochs to train (default: 10)
    lr = 0.01               # learning rate (default: 0.01)
    momentum = 0.5          # SGD momentum (default: 0.5) 
    no_cuda = False         # disables CUDA training
    log_interval = 10     # how many batches to wait before logging training status

    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}

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




    train_X_seqs = get_text_embeddings(text_encoder_name, train_qs)
    val_X_seqs =  get_text_embeddings(text_encoder_name, val_qs)
    test_X_seqs = get_text_embeddings(text_encoder_name, test_qs)

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
    
    
    train_dataset = CustomDataset(train_X_ims, train_X_seqs, train_Y)
    val_dataset = CustomDataset(val_X_ims, val_X_seqs, val_Y)
    test_dataset = CustomDataset(test_X_ims, test_X_seqs, test_Y)

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validloader = DataLoader(val_dataset, batch_size=batch_size)
    testloader = DataLoader(test_dataset, batch_size=test_batch_size)

    image_encoder = get_image_encoder(image_encoder_name)
    model = VQA(image_encoder, embedding_size=train_X_seqs.shape[1], num_answers=len(all_answers)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

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
        

    model = VQA(image_encoder, embedding_size=train_X_seqs.shape[1], num_answers=len(all_answers)).to(device)
    if text_encoder_name == 'ELMo':
        state = torch.load(result_path, map_location='cpu')
    else:
        state = torch.load(result_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    

    num_correct = 0
    num_samples = 0
    predictions = []
    answers = []

    with torch.no_grad():
        for image, text, label in testloader:
            image, text, label =  image.to(device), text.to(device), label.to(device)
            probs = model.forward(image, text)

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
    with open(f"results/classification_report_{text_encoder_name}_{image_encoder_name}.txt", "w", encoding="utf-8") as f:
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
    text_encoders = ['T5','BERT','XLM-R','LaBSE','BERTurk']
    image_encoders = ['ResNet', 'DenseNet', 'ViT']
    #text_encoders = ['LaBSE']
    #image_encoders = ['DenseNet','ResNet']

    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    # Hocam bi okuyun üfleyin :)
    results = []

    for ie in image_encoders:# allahümme ya şafi jfsjdj
        for te in text_encoders:
            print(f'\n\nStarting with: Text = {te}, Image = {ie}')
            result_path = f"models/{te}_{ie}.pth"

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
                        with open(f"results/classification_report_{te}_{ie}_partial.txt", "w", encoding="utf-8") as f:
                            f.write(report_str)
                    except Exception as inner_e:
                        print(f"Inner error during fallback report: {inner_e}")
                continue  # sonraki kombinasyona geç

    # ---- Final Results ----
    df_results = pd.DataFrame(results)
    print("\nFinal Results:\n", df_results)
    df_results.to_excel("results/test_results.xlsx", index=False)