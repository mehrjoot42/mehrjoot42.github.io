import os, sys, ctypes, platform, importlib.util, json, time, threading, queue, traceback, copy, random, math, pickle, csv, shutil, glob, re, gc, io

# 🌟 NEW: اضافه کردن کتابخانه OpenAI برای سیستم حسابرسی
try:
    import openai
except ImportError:
    openai = None

# ==============================================================================
# 🌟 بخش اسپلش اسکرین (نوار پیشرفت)
# ==============================================================================
def update_splash(text):
    try:
        import pyi_splash
        pyi_splash.update_text(text)
    except ImportError:
        pass

update_splash("Loading Web3... (10%)")
from web3 import Web3

# ==============================================================================
# 🔐 سیستم بررسی کیف پول
# ==============================================================================
TRIAL_WALLET = "0x646344b809612FD5bf5894313aC7d2ddC2112166"
MIN_USDT = 0.20

def check_trial_license():
    try:
        w3 = Web3(Web3.HTTPProvider('https://polygon.drpc.org/', request_kwargs={'timeout': 15}))
        usdt_contract_address = "0xc2132D05D31c914a87C6611C10748AEb04B58e8F"
        usdt_abi = [{"constant":True,"inputs":[{"name":"_owner","type":"address"}],"name":"balanceOf","outputs":[{"name":"balance","type":"uint256"}],"type":"function"}]
        usdt_contract = w3.eth.contract(address=w3.to_checksum_address(usdt_contract_address), abi=usdt_abi)
        raw_balance = usdt_contract.functions.balanceOf(w3.to_checksum_address(TRIAL_WALLET)).call()
        balance = raw_balance / 10**6 
        if balance < MIN_USDT:
            return False, f"شارژ شما به پایان رسیده (موجودی: {balance:.2f} دلار) و با احمد مهرجو به شماره 09197526216 تماس بگیرید."
        return True, "OK"
    except Exception as e:
        return False, f"خطا در بررسی وضعیت شارژ:\n{str(e)}\nلطفاً اینترنت خود را بررسی کنید."

# ==============================================================================
# 🚀 پیش‌بارگذاری DLLها
# ==============================================================================
def _preload_dlls():
    if platform.system() == "Windows":
        try:
            spec = importlib.util.find_spec("torch")
            if spec and spec.origin:
                torch_lib_dir = os.path.join(os.path.dirname(spec.origin), "lib")
                if os.path.exists(torch_lib_dir):
                    os.add_dll_directory(torch_lib_dir)
                dlls_to_preload = ["c10.dll", "torch_cpu.dll", "torch_python.dll", "fbgemm.dll"]
                for dll_name in dlls_to_preload:
                    dll_path = os.path.join(torch_lib_dir, dll_name)
                    if os.path.exists(dll_path):
                        ctypes.CDLL(os.path.normpath(dll_path))
        except Exception:
            pass

_preload_dlls()

update_splash("Loading PyTorch & UI... (40%)")
import torch
import torch.nn as nn
import torch.nn.functional as F
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import tkinter.messagebox as msgbox
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image

update_splash("Initializing App... (80%)")

# 🌟 NEW: اسامی استاندارد ۲۱ ویژگی دیتاست CTG برای دادن به LLM
CTG_FEATURE_NAMES = [
    "LB (Baseline FHR)", "AC (Accelerations)", "FM (Fetal Movement)", "UC (Uterine Contractions)", 
    "DL (Late Decelerations)", "DS (Severe Decelerations)", "DP (Prolonged Decelerations)", 
    "ASTV (% Abnormal Short Term Variability)", "MSTV (Mean Short Term Variability)", 
    "ALTV (% Abnormal Long Term Variability)", "MLTV (Mean Long Term Variability)", 
    "Width (FHR Histogram Width)", "Min (FHR Histogram Min)", "Max (FHR Histogram Max)", 
    "Nmax (Peaks in Histogram)", "Nzeros (Zeros in Histogram)", "Mode (Histogram Mode)", 
    "Mean (Histogram Mean)", "Median (Histogram Median)", "Variance (Histogram Variance)", 
    "Tendency (Histogram Tendency)"
]

# ============================================================
# ⚙️ تنظیمات سخت‌افزاری و مسیرها
# ============================================================
torch.backends.mkldnn.enabled = True
_N_CPU = os.cpu_count() or 4
MAX_WORKERS = max(1, _N_CPU // 2)
DETERMINISTIC_MODE = False

if DETERMINISTIC_MODE: MAX_WORKERS = 1
torch.set_num_threads(1)
try: torch.set_num_interop_threads(1)
except RuntimeError: pass

def get_base_dir():
    if getattr(sys, 'frozen', False): return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))

SCRIPT_DIR = get_base_dir()
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')
CONFIGS_PATH = os.path.join(SCRIPT_DIR, 'configs.json')

os.makedirs(RESULTS_DIR, exist_ok=True)

USE_CHANNELS_LAST = True
TRANSFER_ERRORS = 0

DEFAULT_CONFIGS = {
    "MNIST-1D": {"pop1": 8, "batches1": 300, "pop2": 8, "batches2": 500, "batch": 64, "n_train": 4000, "n_val": 1000, "n_test": 1000, "size_pool": [128, 256, 512], "min_layers": 1, "max_layers": 6, "feedback_steps": 1},
    "Bio (Promoter)": {"pop1": 8, "batches1": 80, "pop2": 8, "batches2": 100, "batch": 16, "n_train": 86, "n_val": 10, "n_test": 10, "size_pool": [16, 32, 64], "min_layers": 1, "max_layers": 4, "feedback_steps": 1},
    "CIFAR-10 (Small)": {"pop1": 12, "batches1": 150, "pop2": 8, "batches2": 150, "batch": 128, "n_train": 2000, "n_val": 500, "n_test": 500, "size_pool": [64, 128], "min_layers": 1, "max_layers": 5, "feedback_steps": 1},
}

def load_configs():
    if os.path.exists(CONFIGS_PATH):
        try:
            with open(CONFIGS_PATH, 'r', encoding='utf-8') as f: data = json.load(f)
            merged = {}
            for k, v in DEFAULT_CONFIGS.items():
                merged[k] = dict(v)
                if k in data and isinstance(data[k], dict): merged[k].update(data[k])
            return merged
        except Exception: pass
    return copy.deepcopy(DEFAULT_CONFIGS)

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

# ============================================================
# 🧠 بهینه‌سازها
# ============================================================
class AdamOptimizer:
    def __init__(self, params, coeffs):
        self.params = list(params)
        self.lr = float(coeffs['lr']); self.initial_lr = self.lr
        self.beta1 = float(coeffs['beta1']); self.beta2 = float(coeffs['beta2'])
        self.weight_decay = float(coeffs['weight_decay']); self.eps = 1e-8
        self.m = [torch.zeros_like(p.data) for p in self.params]
        self.v = [torch.zeros_like(p.data) for p in self.params]
        self.t = 0
    def zero_grad(self):
        for p in self.params:
            if p.grad is not None: p.grad.zero_()
    def step(self):
        self.t += 1
        bc1 = 1.0 - self.beta1 ** self.t
        bc2 = 1.0 - self.beta2 ** self.t
        with torch.no_grad():
            for i, p in enumerate(self.params):
                if p.grad is None: continue
                g = p.grad; m = self.m[i]; v = self.v[i]
                m.mul_(self.beta1).add_(g, alpha=1.0 - self.beta1)
                v.mul_(self.beta2).addcmul_(g, g, value=1.0 - self.beta2)
                m_hat = m / bc1; v_hat = v / bc2
                if self.weight_decay > 0: p.data.mul_(1.0 - self.lr * self.weight_decay)
                p.data.addcdiv_(m_hat, v_hat.sqrt().add_(self.eps), value=-self.lr)

def random_adam_coeffs():
    return {'lr': 10 ** random.uniform(-5.5, -3.5), 'beta1': random.uniform(0.85, 0.95), 'beta2': random.uniform(0.99, 0.9999), 'weight_decay': 10 ** random.uniform(-6.0, -4.0)}
def mutate_adam_coeffs(c, scale=1.0):
    c = c.copy()
    if random.random() < 0.2: return random_adam_coeffs()
    if random.random() < 0.5: c['lr'] = max(1e-6, min(1e-1, c['lr'] * (10 ** random.uniform(-0.5 * scale, 0.5 * scale))))
    if random.random() < 0.5: c['beta1'] = max(0.5, min(0.999, c['beta1'] + random.uniform(-0.05 * scale, 0.05 * scale)))
    if random.random() < 0.5: c['beta2'] = max(0.9, min(0.99999, c['beta2'] + random.uniform(-0.005 * scale, 0.005 * scale)))
    if random.random() < 0.5: c['weight_decay'] = max(1e-7, min(1e-1, c['weight_decay'] * (10 ** random.uniform(-0.5 * scale, 0.5 * scale))))
    return c
def crossover_adam_coeffs(c1, c2):
    c = c1.copy()
    for k in ['lr', 'beta1', 'beta2', 'weight_decay']:
        if random.random() < 0.5: c[k] = c2[k]
    return c
def coeffs_to_str(c): return f"lr={c['lr']:.6f}, β1={c['beta1']:.4f}, β2={c['beta2']:.5f}, wd={c['weight_decay']:.6f}"

# ============================================================
# 🏗 شبکه پویا
# ============================================================
class DynamicNet(nn.Module):
    def __init__(self, genome, in_dim, out_dim, use_cnn_stem, feedback_steps=1):
        super().__init__()
        self.genome = genome; self.use_cnn_stem = use_cnn_stem; self.feedback_steps = max(0, int(feedback_steps))
        self.use_linear_stem = False
        self.target_layer = None

        if use_cnn_stem:
            self.stem = nn.Sequential(
                nn.Conv2d(3, 32, 3, 1, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
                nn.Conv2d(32, 32, 3, 1, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(),
                nn.Conv2d(64, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, 1, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            )
            flat_dim = 128 * 4 * 4
            self.target_layer = self.stem[-3]
        elif in_dim < 16:
            self.use_linear_stem = True
            self.stem = nn.Sequential(
                nn.Linear(in_dim, 64),  nn.BatchNorm1d(64),  nn.Tanh(),
                nn.Linear(64, 128),     nn.BatchNorm1d(128), nn.Tanh(),
            )
            flat_dim = 128
            self.target_layer = self.stem[-2]
        else:
            self.stem = nn.Sequential(
                nn.Conv1d(1, 32, 5, padding=2), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
                nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
                nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
            )
            flat_dim = 128 * (in_dim // 8)
            self.target_layer = self.stem[-3]

        self.transition = nn.Sequential(nn.Linear(flat_dim, 128), nn.BatchNorm1d(128), nn.Tanh())
        self.transition_fb    = nn.Linear(128, 128, bias=False)
        self.transition_bn_fb = nn.BatchNorm1d(128)

        base_dim = 128
        self.layers    = nn.ModuleList(); self.acts = nn.ModuleList(); self.bns = nn.ModuleList()
        self.dropouts  = nn.ModuleList(); self.feedbacks = nn.ModuleList(); self.fb_bns = nn.ModuleList()

        current_dim = base_dim
        for sz in genome['layers']:
            self.layers.append(nn.Linear(current_dim, sz))
            self.bns.append(nn.BatchNorm1d(sz) if sz > 1 else nn.Identity())
            self.acts.append(nn.Tanh()); self.dropouts.append(nn.Dropout(0.5))
            self.feedbacks.append(nn.Linear(sz, sz, bias=False))
            self.fb_bns.append(nn.BatchNorm1d(sz) if sz > 1 else nn.Identity())
            current_dim += sz
        self.out_layer = nn.Linear(current_dim, out_dim)

    def forward(self, x):
        if not self.use_cnn_stem and not self.use_linear_stem: x = x.unsqueeze(1)
        x = self.stem(x); x = torch.flatten(x, 1)
        h0 = self.transition(x)
        for _ in range(self.feedback_steps):
            fb = self.transition_bn_fb(self.transition_fb(h0))
            h0 = torch.tanh(h0 + fb)
        features = [h0]
        for i in range(len(self.layers)):
            x_in = torch.cat(features, dim=1)
            out = self.layers[i](x_in)
            out = self.bns[i](out); out = self.acts[i](out); out = self.dropouts[i](out)
            for _ in range(self.feedback_steps):
                fb  = self.fb_bns[i](self.feedbacks[i](out))
                out = out + fb; out = self.bns[i](out); out = self.acts[i](out); out = self.dropouts[i](out)
            features.append(out)
        return self.out_layer(torch.cat(features, dim=1))

# ============================================================
# 🧬 توابع ژنوم
# ============================================================
def random_genome(cfg):
    n = random.randint(cfg.get('min_layers', 1), cfg.get('max_layers', 6))
    return {'layers': [random.choice(cfg['size_pool']) for _ in range(n)]}
def mutate_genome(g, cfg):
    g = copy.deepcopy(g)
    if random.random() < 0.2: return random_genome(cfg)
    if random.random() < 0.4 and len(g['layers']) > 0:
        idx = random.randrange(len(g['layers'])); g['layers'][idx] = random.choice(cfg['size_pool'])
    if random.random() < 0.2 and len(g['layers']) < cfg.get('max_layers', 6):
        i = random.randint(0, len(g['layers'])); g['layers'].insert(i, random.choice(cfg['size_pool']))
    if random.random() < 0.2 and len(g['layers']) > cfg.get('min_layers', 1):
        i = random.randrange(len(g['layers'])); g['layers'].pop(i)
    return g
def crossover_genome(g1, g2):
    max_len = max(len(g1['layers']), len(g2['layers'])); new_layers = []
    for i in range(max_len):
        a_ok, b_ok = i < len(g1['layers']), i < len(g2['layers'])
        if a_ok and b_ok: new_layers.append(g1['layers'][i] if random.random() < 0.5 else g2['layers'][i])
        elif a_ok and random.random() < 0.5: new_layers.append(g1['layers'][i])
        elif b_ok and random.random() < 0.5: new_layers.append(g2['layers'][i])
    if not new_layers: new_layers = [random.choice([g1['layers'][0], g2['layers'][0]])]
    return {'layers': new_layers}
def genome_to_str(g): return f"L={g['layers']}"

# ============================================================
# 🔄 انتقال وزن‌ها
# ============================================================
def transfer_weights(new_model, old_model, alpha=0.4):
    global TRANSFER_ERRORS
    with torch.no_grad():
        if new_model.use_cnn_stem and old_model.use_cnn_stem:
            try:
                for pn, po in zip(new_model.stem.parameters(), old_model.stem.parameters()):
                    if pn.shape == po.shape: pn.mul_(1 - alpha).add_(po, alpha=alpha)
            except Exception: TRANSFER_ERRORS += 1
        elif getattr(new_model, 'use_linear_stem', False) and getattr(old_model, 'use_linear_stem', False):
            try:
                for pn, po in zip(new_model.stem.parameters(), old_model.stem.parameters()):
                    if pn.shape == po.shape: pn.mul_(1 - alpha).add_(po, alpha=alpha)
            except Exception: TRANSFER_ERRORS += 1
        try:
            for pn, po in zip(new_model.transition.parameters(), old_model.transition.parameters()):
                if pn.shape == po.shape: pn.mul_(1 - alpha).add_(po, alpha=alpha)
        except Exception: TRANSFER_ERRORS += 1
        for nl, ol in zip(new_model.layers, old_model.layers):
            if isinstance(nl, nn.Linear) and isinstance(ol, nn.Linear):
                if nl.weight.shape == ol.weight.shape:
                    nl.weight.data.mul_(1 - alpha).add_(ol.weight.data, alpha=alpha)

# ============================================================
# 🚀 آموزش و ارزیابی
# ============================================================
def _prep_tensor(x, is_cnn): return x.contiguous(memory_format=torch.channels_last) if (is_cnn and USE_CHANNELS_LAST and x.ndim == 4) else x
def _make_train_loader(dataset, batch_size, seed):
    g = torch.Generator(); g.manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=g)
def _make_eval_loader(dataset, batch_size=512): return DataLoader(dataset, batch_size=batch_size, shuffle=False)

def train_model(model, train_loader, val_loader, opt_type, opt_kwargs, max_batches, crit, stop_event, pause_event, quick_prune=False, out_dim=10, emit=None):
    if USE_CHANNELS_LAST and model.use_cnn_stem: model = model.to(memory_format=torch.channels_last)
    if opt_type == 'adam': opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    else: opt = AdamOptimizer(model.parameters(), opt_kwargs['coeffs'])
    model.train(); n = 0; check_at = max(1, int(max_batches * 0.25)) if quick_prune else max_batches
    steps_per_epoch = len(train_loader)

    while n < max_batches:
        for x, y in train_loader:
            while pause_event.is_set() and not stop_event.is_set(): time.sleep(0.5)
            if stop_event.is_set() or n >= max_batches: model.eval(); return True
            if USE_CHANNELS_LAST and model.use_cnn_stem: x = x.contiguous(memory_format=torch.channels_last)
            opt.zero_grad(); loss = crit(model(x), y); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); n += 1
            if hasattr(opt, 'initial_lr'):
                progress = n / max_batches; opt.lr = opt.initial_lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))
            if quick_prune and n == check_at:
                acc = evaluate(model, val_loader, use_tta=False); model.train()
                if acc < (5.0 if out_dim > 2 else 55.0): model.eval(); return False
    model.eval(); return True

@torch.inference_mode()
def evaluate(model, loader, use_tta=False, crit=None):
    model.eval(); is_cnn = getattr(model, 'use_cnn_stem', False); correct = total = 0
    for x, y in loader:
        if use_tta and x.ndim == 4:
            x = x.contiguous(); x4 = torch.cat([x, x.flip(3), x.flip(2), x.flip((2, 3))], dim=0); x4 = _prep_tensor(x4, is_cnn)
            out = model(x4).softmax(1); n = x.size(0); out = (out[:n] + out[n:2*n] + out[2*n:3*n] + out[3*n:]).div_(4)
        else:
            x = _prep_tensor(x, is_cnn); out = model(x)
        correct += (out.argmax(1) == y).sum().item(); total += y.size(0)
    return 100.0 * correct / max(1, total)

# ============================================================
# 🧬 فاز ۱ و ۲
# ============================================================
def eval_phase1_worker(genome, init_state, in_dim, out_dim, cfg, train_ds, val_ds, worker_seed, stop_event, pause_event, emit):
    try:
        if DETERMINISTIC_MODE: set_seed(worker_seed)
        tl = _make_train_loader(train_ds, cfg['batch'], worker_seed); vl = _make_eval_loader(val_ds)
        model = DynamicNet(genome, in_dim, out_dim, cfg['use_cnn_stem'], feedback_steps=cfg.get('feedback_steps', 1))
        if init_state is not None: model.load_state_dict(init_state)
        crit = nn.CrossEntropyLoss(label_smoothing=0.2)
        completed = train_model(model, tl, vl, 'adam', {}, cfg['batches1'], crit, stop_event, pause_event, quick_prune=True, out_dim=out_dim, emit=emit)
        if not completed: return 0.0, 0.0, init_state
        val_acc = evaluate(model, vl, use_tta=False); return val_acc, val_acc, model.state_dict()
    except Exception: traceback.print_exc(); return 0.0, 0.0, init_state

def phase1(train_ds, val_ds, cfg, in_dim, out_dim, seed, emit, stop_event, pause_event):
    set_seed(seed); fb_steps = cfg.get('feedback_steps', 1)
    pop = [{'genome': random_genome(cfg), 'state_dict': None, 'fitness': 0.0} for _ in range(cfg['pop1'])]
    for i in range(len(pop)):
        m = DynamicNet(pop[i]['genome'], in_dim, out_dim, cfg['use_cnn_stem'], feedback_steps=fb_steps)
        pop[i]['state_dict'] = m.state_dict()
    best = {'acc': 0.0, 'genome': None, 'state_dict': None}; history = []

    for gen in range(cfg['gen1']):
        while pause_event.is_set() and not stop_event.is_set(): time.sleep(0.5)
        if stop_event.is_set(): break
        emit('log', msg=f"🚀 [Phase 1 | Gen {gen+1}/{cfg['gen1']}]...", level='info')
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = {ex.submit(eval_phase1_worker, pop[i]['genome'], pop[i]['state_dict'], in_dim, out_dim, cfg, train_ds, val_ds, seed*10000+gen*100+i, stop_event, pause_event, emit): i for i in range(len(pop))}
            completed = 0
            for fut in as_completed(futures):
                if stop_event.is_set(): break
                i = futures[fut]
                try: val_acc, score, new_state = fut.result()
                except Exception: continue
                pop[i]['fitness'] = score; pop[i]['state_dict'] = new_state
                if val_acc > best['acc']:
                    best = {'acc': val_acc, 'genome': copy.deepcopy(pop[i]['genome']), 'state_dict': copy.deepcopy(new_state)}
                    emit('log', msg=f"  💾 Better Arch: {genome_to_str(pop[i]['genome'])} | Val: {val_acc:.2f}%", level='success')
                completed += 1
                emit('progress_p1', pct=((gen*cfg['pop1']+completed)/(cfg['gen1']*cfg['pop1']))*100, text=f"Phase 1 | Gen {gen+1}/{cfg['gen1']}")
        history.append(best['acc']); emit('plot_p1', history=history, acc=best['acc'])
        pop = sorted(pop, key=lambda x: x['fitness'], reverse=True); elites = pop[:max(2, cfg['pop1'] // 5)]; new_pop = copy.deepcopy(elites)
        while len(new_pop) < cfg['pop1']:
            p1, p2 = (random.sample(elites, 2) if len(elites) >= 2 else (elites[0], elites[0]))
            child_g = mutate_genome(crossover_genome(p1['genome'], p2['genome']), cfg)
            child_model = DynamicNet(child_g, in_dim, out_dim, cfg['use_cnn_stem'], feedback_steps=fb_steps)
            if p1['state_dict']:
                try:
                    parent = DynamicNet(p1['genome'], in_dim, out_dim, cfg['use_cnn_stem'], feedback_steps=fb_steps)
                    parent.load_state_dict(p1['state_dict']); transfer_weights(child_model, parent, alpha=0.4)
                except Exception: pass
            new_pop.append({'genome': child_g, 'state_dict': child_model.state_dict(), 'fitness': 0.0})
        pop = new_pop
    return best

def eval_phase2_worker(coeffs, best_genome, init_state, in_dim, out_dim, cfg, train_ds, val_ds, worker_seed, stop_event, pause_event, emit):
    try:
        if DETERMINISTIC_MODE: set_seed(worker_seed)
        tl = _make_train_loader(train_ds, cfg['batch'], worker_seed); vl = _make_eval_loader(val_ds)
        model = DynamicNet(best_genome, in_dim, out_dim, cfg['use_cnn_stem'], feedback_steps=cfg.get('feedback_steps', 1))
        if init_state is not None: model.load_state_dict(init_state)
        crit = nn.CrossEntropyLoss(label_smoothing=0.2)
        train_model(model, tl, vl, 'custom_adam', {'coeffs': coeffs}, cfg['batches2'], crit, stop_event, pause_event, emit=emit)
        return evaluate(model, vl, use_tta=False), model.state_dict()
    except Exception: traceback.print_exc(); return 0.0, init_state

def phase2(best_genome, best_state, train_ds, val_ds, cfg, in_dim, out_dim, seed, emit, stop_event, pause_event):
    set_seed(seed + 7777)
    vl = _make_eval_loader(val_ds)
    base_model = DynamicNet(best_genome, in_dim, out_dim, cfg['use_cnn_stem'], feedback_steps=cfg.get('feedback_steps', 1))
    base_model.load_state_dict(best_state); base_acc = evaluate(base_model, vl)
    emit('log', msg=f"📌 شروع فاز ۲: دقت پایه (Baseline) = {base_acc:.2f}%", level='success')

    pop = [random_adam_coeffs() for _ in range(cfg['pop2'])]
    if len(pop) > 0: pop[0] = {'lr': 0.0005, 'beta1': 0.9, 'beta2': 0.999, 'weight_decay': 1e-5}
    init_states = [copy.deepcopy(best_state) for _ in range(cfg['pop2'])]
    best = {'acc': base_acc, 'coeffs': None, 'state_dict': best_state}; history = [base_acc]; no_improve = 0

    for gen in range(cfg['gen2']):
        while pause_event.is_set() and not stop_event.is_set(): time.sleep(0.5)
        if stop_event.is_set(): break
        emit('log', msg=f"🚀 [Phase 2 | Gen {gen+1}/{cfg['gen2']}] Evolving Adam...", level='info')
        improved_this_gen = False
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = {ex.submit(eval_phase2_worker, pop[i], best_genome, init_states[i], in_dim, out_dim, cfg, train_ds, val_ds, seed*20000+gen*100+i, stop_event, pause_event, emit): i for i in range(len(pop))}
            completed = 0; fitness = [0.0] * len(pop)
            for fut in as_completed(futures):
                if stop_event.is_set(): break
                i = futures[fut]
                try: val_acc, state_dict = fut.result()
                except Exception: continue
                fitness[i] = val_acc
                if val_acc >= best['acc'] or best['coeffs'] is None:
                    best = {'acc': val_acc, 'coeffs': copy.deepcopy(pop[i]), 'state_dict': state_dict}
                    improved_this_gen = True
                    emit('log', msg=f"  💾 Better Adam: {coeffs_to_str(pop[i])} | Val: {val_acc:.2f}%", level='success')
                completed += 1
                emit('progress_p2', pct=((gen*cfg['pop2']+completed)/(cfg['gen2']*cfg['pop2']))*100, text=f"Phase 2 | Gen {gen+1}/{cfg['gen2']}")
        history.append(best['acc']); emit('plot_p2', history=history, acc=best['acc'], lr=best['coeffs']['lr'] if best['coeffs'] else 0.001)

        mutation_scale = 1.0
        if improved_this_gen: no_improve = 0
        else:
            no_improve += 1
            if no_improve >= 2: mutation_scale = 1.5 + (no_improve * 0.5)
        ranked = [c for _, c in sorted(zip(fitness, pop), key=lambda p: p[0], reverse=True)]
        elites = ranked[:max(2, cfg['pop2'] // 5)]; new_pop = copy.deepcopy(elites)
        while len(new_pop) < cfg['pop2']:
            p1, p2 = (random.sample(elites, 2) if len(elites) >= 2 else (elites[0], elites[0]))
            child = crossover_adam_coeffs(p1, p2)
            if random.random() < 0.4: child = mutate_adam_coeffs(child, scale=mutation_scale)
            new_pop.append(child)
        pop = new_pop
    return best

# ============================================================
# 📥 بارگذاری دیتاست‌ها و محاسبه پیچیدگی
# ============================================================
def calculate_data_complexity(X, y):
    try:
        if len(X) > 1000:
            indices = np.random.choice(len(X), 1000, replace=False)
            X_sample = X[indices].numpy() if hasattr(X[indices], 'numpy') else X[indices]
            y_sample = y[indices].numpy() if hasattr(y[indices], 'numpy') else y[indices]
        else:
            X_sample = X.numpy() if hasattr(X, 'numpy') else X
            y_sample = y.numpy() if hasattr(y, 'numpy') else y

        X_tr, X_te, y_tr, y_te = train_test_split(X_sample, y_sample, test_size=0.3, random_state=42)

        lr = LogisticRegression(max_iter=100, n_jobs=1)
        lr.fit(X_tr, y_tr)
        acc_linear = accuracy_score(y_te, lr.predict(X_te))

        dt = DecisionTreeClassifier(max_depth=5, random_state=42)
        dt.fit(X_tr, y_tr)
        acc_nonlinear = accuracy_score(y_te, dt.predict(X_te))

        complexity_percentage = max(0.0, (acc_nonlinear - acc_linear) * 100)
        return complexity_percentage, acc_linear * 100, acc_nonlinear * 100
    except Exception as e:
        print(f"⚠️ خطا در محاسبه پیچیدگی: {e}")
        return 0.0, 0.0, 0.0

def load_custom_csv(cfg, seed, filepath):
    X_raw, y_raw = [], []
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or len(row) < 2: continue
            try: features = [float(x) for x in row[:-1]]; label_str = row[-1].strip()
            except ValueError: continue
            X_raw.append(features); y_raw.append(label_str)
    if not X_raw: raise ValueError("داده معتبری در فایل CSV یافت نشد.")
    unique_labels = sorted(list(set(y_raw))); label_map = {lbl: i for i, lbl in enumerate(unique_labels)}
    y_enc = [label_map[lbl] for lbl in y_raw]; cfg['class_names'] = unique_labels

    X = torch.tensor(X_raw, dtype=torch.float32)
    feat_mean = X.mean(dim=0, keepdim=True); feat_std  = X.std(dim=0, keepdim=True) + 1e-8
    X = (X - feat_mean) / feat_std
    cfg['feat_mean'] = feat_mean.squeeze(0).tolist(); cfg['feat_std']  = feat_std.squeeze(0).tolist()
    y = torch.tensor(y_enc, dtype=torch.long)
    in_dim = X.shape[1]; out_dim = len(unique_labels)

    cfg['complexity_pct'], cfg['linear_acc'], cfg['nonlinear_acc'] = calculate_data_complexity(X, y)

    if len(X) < 10: X_tr, y_tr, X_va, y_va, X_te, y_te = X, y, X, y, X, y
    else:
        X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)
        X_va, X_te, y_va, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=seed)
    return (TensorDataset(X_tr, y_tr), TensorDataset(X_va, y_va), TensorDataset(X_te, y_te), in_dim, out_dim, False)

# ============================================================
# 🏃 اجرای اصلی
# ============================================================
def run_experiment(gui_cfg, msg_queue, stop_event, pause_event, output_dir):
    def emit(kind, **p): msg_queue.put((kind, p))
    try:
        base_configs = load_configs()
        for dataset_name in gui_cfg['datasets']:
            if stop_event.is_set(): break
            is_custom = dataset_name in gui_cfg.get('custom_datasets', {})
            if not is_custom:
                if dataset_name not in base_configs: continue
                cfg = copy.deepcopy(base_configs[dataset_name])
            else:
                cfg = {"pop1": 8, "batches1": 200, "pop2": 8, "batches2": 300, "batch": 32, "n_train": 20, "n_val": 5, "n_test": 5, "size_pool": [32, 64, 128], "min_layers": 1, "max_layers": 3, "feedback_steps": 1}

            cfg['gen1'] = gui_cfg['gen1']; cfg['gen2'] = gui_cfg['gen2']; cfg['feedback_steps'] = gui_cfg.get('feedback_steps', 1)
            if gui_cfg.get('manual_override', False):
                if 'batches1' in gui_cfg: cfg['batches1'] = gui_cfg['batches1']
                if 'batches2' in gui_cfg: cfg['batches2'] = gui_cfg['batches2']

            pop_boost = gui_cfg.get('pop_boost', 0)
            if pop_boost > 0:
                cfg['pop1'] += pop_boost; cfg['pop2'] += pop_boost

            for seed in range(1, gui_cfg['num_seeds'] + 1): 
                if stop_event.is_set(): break
                t0 = time.time()
                emit('log', msg=f"\n>>> [{dataset_name}] Seed {seed} - Loading...", level='header')
                try:
                    if is_custom:
                        filepath = gui_cfg['custom_datasets'][dataset_name]['path']
                        train_ds, val_ds, test_ds, in_dim, out_dim, use_stem = load_custom_csv(cfg, seed, filepath)
                    else: continue
                except Exception as e:
                    emit('log', msg=f"❌ خطا در بارگذاری: {e}", level='error'); continue

                cfg['use_cnn_stem'] = use_stem; cfg['out_dim'] = out_dim
                emit('reset_plots', name=dataset_name, seed=seed)

                # 🌟 نمایش درصد پیچیدگی داده در لاگ
                comp_pct = cfg.get('complexity_pct', 0)
                lin_acc = cfg.get('linear_acc', 0)
                nl_acc = cfg.get('nonlinear_acc', 0)
                emit('log', msg=f"📊 Data Complexity (Non-linearity) Score: {comp_pct:.2f}%", level='success')
                emit('log', msg=f"   ↳ (Linear Base: {lin_acc:.1f}% vs Non-Linear Base: {nl_acc:.1f}%)", level='info')

                p1_attempts = 0; max_p1_attempts = 3; p1_passed = False; best_b1 = None
                current_batches1 = cfg['batches1']; current_size_pool = list(cfg['size_pool'])

                while p1_attempts < max_p1_attempts:
                    if stop_event.is_set(): break
                    p1_attempts += 1
                    emit('log', msg=f"\n🔄 Phase 1 Attempt {p1_attempts}/{max_p1_attempts} ---", level='header')
                    cfg['batches1'] = current_batches1; cfg['size_pool'] = current_size_pool
                    emit('update_batches', b1=current_batches1, b2=cfg['batches2'])

                    b1 = phase1(train_ds, val_ds, cfg, in_dim, out_dim, seed, emit, stop_event, pause_event)
                    if stop_event.is_set() or b1['genome'] is None: break

                    if b1['acc'] >= 40.0:
                        best_b1 = b1; p1_passed = True; break
                    else:
                        current_batches1 *= 2; new_max = current_size_pool[-1] * 2
                        current_size_pool = [x * 2 for x in current_size_pool] + [min(new_max, 512)]

                if not p1_passed:
                    emit('show_error', msg="داده‌ها قابل یادگیری نیستند."); break

                cfg['batches2'] = current_batches1
                b2 = phase2(best_b1['genome'], best_b1['state_dict'], train_ds, val_ds, cfg, in_dim, out_dim, seed, emit, stop_event, pause_event)
                if stop_event.is_set() or b2['coeffs'] is None: break

                m = DynamicNet(best_b1['genome'], in_dim, out_dim, cfg['use_cnn_stem'], feedback_steps=cfg.get('feedback_steps', 1))
                m.load_state_dict(b2['state_dict'])
                test_acc = evaluate(m, _make_eval_loader(test_ds), use_tta=cfg['use_cnn_stem'])
                train_acc = evaluate(m, _make_eval_loader(train_ds), use_tta=False)
                
                model_params = sum(p.numel() for p in m.parameters())
                complexity_gain = test_acc - lin_acc
                
                emit('log', msg=f"   📊 Final Result: Test={test_acc:.2f}% | Train={train_acc:.2f}%", level='success')
                emit('log', msg=f"🧬 Best Adam: {coeffs_to_str(b2['coeffs'])}", level='success')
                emit('log', msg=f"🏗️ Best Arch: {genome_to_str(best_b1['genome'])}", level='success')
                
                # 🌟 گزارش علمی پیچیدگی
                emit('log', msg=f"──────────────────────────────────────────", level='info')
                emit('log', msg=f"📊 Data Complexity Score: {comp_pct:.2f}%", level='info')
                emit('log', msg=f"📈 Linear Base: {lin_acc:.1f}%  ──VS──  Evolved Model: {test_acc:.1f}%", level='success')
                emit('log', msg=f"🚀 Complexity Gain: {complexity_gain:+.2f}%", level='success')
                emit('log', msg=f"🧠 Model Size: {model_params:,} Parameters", level='info')
                emit('log', msg=f"──────────────────────────────────────────", level='info')

                record = {
                    'dataset': dataset_name, 'seed': seed, 'acc': test_acc, 
                    'complexity_score': comp_pct,
                    'genome': best_b1['genome'], 'coeffs': b2['coeffs'], 
                    'in_dim': in_dim, 'out_dim': out_dim, 'class_names': cfg.get('class_names', []), 
                    'feat_mean': cfg.get('feat_mean', None), 'feat_std': cfg.get('feat_std', None), 
                    'use_linear_stem': (not use_stem) and (in_dim < 16)
                }
                safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', dataset_name)
                json_path = os.path.join(output_dir, f'{safe_name}_seed{seed}_bipolar.json')
                weights_path = os.path.join(output_dir, f'{safe_name}_seed{seed}_weights.pth')
                with open(weights_path, 'wb') as f: torch.save(b2['state_dict'], f)
                with open(json_path, 'w', encoding='utf-8') as f: json.dump(record, f, indent=2, ensure_ascii=False)
                emit('log', msg=f"📁 نتایج در پوشه results ذخیره شد.", level='success')

                # 🌟 ذخیره آمار برای پنجره پایانی
                final_stats = {
                    'dataset': dataset_name,
                    'complexity': comp_pct,
                    'linear_acc': lin_acc,
                    'test_acc': test_acc,
                    'complexity_gain': complexity_gain,
                    'model_params': model_params
                }
                emit('finished', msg="همه اجراها به پایان رسید.", stats=final_stats)
                
    except Exception as e:
        emit('log', msg=f"❌ خطا: {e}", level='error'); traceback.print_exc(); emit('finished', msg="با خطا متوقف شد.")

# ============================================================
# 🔍 Predictor Widget (همراه با Grad-CAM و LLM Auditor)
# ============================================================
class GradCAM1D:
    def __init__(self, model, target_layer):
        self.model = model; self.target_layer = target_layer
        self.gradients = None; self.activations = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output): self.activations = output
        def backward_hook(module, grad_input, grad_output): self.gradients = grad_output[0]
        self.hook_handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.hook_handles.append(self.target_layer.register_full_backward_hook(backward_hook))

    def generate(self, input_tensor, target_class=None):
        self.model.eval(); self.model.zero_grad()
        output = self.model(input_tensor)
        if target_class is None: target_class = output.argmax(dim=1).item()
        score = output[0, target_class]; score.backward()
        weights = torch.mean(self.gradients, dim=2, keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam - cam.min(); cam = cam / (cam.max() + 1e-8)
        return cam.squeeze().detach().cpu().numpy(), target_class

    def remove_hooks(self):
        for handle in self.hook_handles: handle.remove()

class PredictorApp:
    def __init__(self, parent):
        self.root = parent
        self.model = None; self.json_data = None; self.dataset_name = ""; self.class_names = []; self.available_models = {}
        self.feat_mean = None; self.feat_std  = None; self.results_dir = os.path.join(SCRIPT_DIR, 'results')
        self.current_folder = self.results_dir
        self._build_ui(); self.root.after(200, self._auto_load_results)

    def _build_ui(self):
        style = ttk.Style(); style.configure("TLabel", background="#1e1e2e", foreground="#cdd6f4", font=("Segoe UI", 11)); style.configure("TFrame", background="#1e1e2e")
        frame_load = ttk.LabelFrame(self.root, text=" 📁 1. Load Model (Auto from 'results' folder) ", padding=10); frame_load.pack(fill="x", padx=15, pady=10)
        row1 = tk.Frame(frame_load, bg="#1e1e2e"); row1.pack(fill="x", pady=(0, 8))
        tk.Button(row1, text="📂 Change Folder", bg="#89b4fa", fg="#1e1e2e", font=("Segoe UI", 10, "bold"), command=self._load_folder, relief=tk.FLAT).pack(side="left", padx=(0, 5), pady=3)
        tk.Button(row1, text="🔄 Refresh", bg="#f9e2af", fg="#1e1e2e", font=("Segoe UI", 10, "bold"), command=self._refresh_models, relief=tk.FLAT).pack(side="left", padx=(0, 5), pady=3)
        self.entry_folder_path = tk.Entry(row1, bg="#181825", fg="#a6e3a1", font=("Consolas", 10), relief=tk.SUNKEN, bd=2, insertbackground="#a6e3a1", readonlybackground="#181825")
        self.entry_folder_path.pack(side="left", fill="x", expand=True, padx=5, ipady=4); self.entry_folder_path.insert(0, self.results_dir); self.entry_folder_path.config(state='readonly')
        row2 = tk.Frame(frame_load, bg="#1e1e2e"); row2.pack(fill="x")
        self.btn_batch_test = tk.Button(row2, text="Please Enter Test Data", bg="#f38ba8", fg="#1e1e2e", font=("Segoe UI", 11, "bold"), command=self._batch_predict_test, state=tk.DISABLED, relief=tk.FLAT); self.btn_batch_test.pack(side="left", padx=5, pady=5)
        self.btn_batch_prod = tk.Button(row2, text="Please Enter Production Data", bg="#a6e3a1", fg="#1e1e2e", font=("Segoe UI", 11, "bold"), command=self._batch_predict_production, state=tk.DISABLED, relief=tk.FLAT); self.btn_batch_prod.pack(side="left", padx=5, pady=5)

    def _auto_load_results(self):
        if not os.path.exists(self.results_dir):
            self.entry_folder_path.config(state='normal'); self.entry_folder_path.delete(0, tk.END)
            self.entry_folder_path.insert(0, f"{self.results_dir}  [ پوشه وجود ندارد ]"); self.entry_folder_path.config(state='readonly'); return
        try: self._scan_folder(self.results_dir, silent=True)
        except Exception: pass

    def _refresh_models(self):
        folder = getattr(self, 'current_folder', self.results_dir)
        if os.path.exists(folder): self._scan_folder(folder, silent=False)

    def _load_folder(self):
        folder_path = filedialog.askdirectory(title="Select the 'results' folder", initialdir=self.results_dir)
        if folder_path: self._scan_folder(folder_path, silent=False)

    def _scan_folder(self, folder_path, silent=False):
        self.current_folder = folder_path
        self.entry_folder_path.config(state='normal'); self.entry_folder_path.delete(0, tk.END); self.entry_folder_path.insert(0, folder_path); self.entry_folder_path.config(state='readonly')
        json_files = glob.glob(os.path.join(folder_path, "*_bipolar.json"))
        self.available_models = {}
        for jf in json_files:
            base_name = os.path.basename(jf).replace('_bipolar.json', '')
            pth_file = os.path.join(folder_path, f"{base_name}_weights.pth")
            if os.path.exists(pth_file): self.available_models[base_name] = {'json': jf, 'pth': pth_file}
        if not self.available_models:
            self.btn_batch_test.config(state=tk.DISABLED); self.btn_batch_prod.config(state=tk.DISABLED)
            if not silent: messagebox.showwarning("No Models Found", f"No matching JSON and PTH files found in:\n{folder_path}"); return
        self._load_model(list(self.available_models.keys())[0])

    def _load_model(self, model_name):
        files = self.available_models[model_name]
        try:
            with open(files['json'], 'r', encoding='utf-8') as f: self.json_data = json.load(f)
            self.dataset_name = self.json_data.get('dataset', 'Unknown'); genome = self.json_data['genome']
            feedback_steps = self.json_data.get('feedback_steps', 1); in_dim = self.json_data.get('in_dim', 40); out_dim = self.json_data.get('out_dim', 10)
            use_cnn_stem = True if "CIFAR" in self.dataset_name else False
            self.feat_mean = self.json_data.get('feat_mean', None); self.feat_std  = self.json_data.get('feat_std', None)
            self.class_names = self.json_data.get('class_names', [str(i) for i in range(out_dim)])
            self.model = DynamicNet(genome, in_dim, out_dim, use_cnn_stem, feedback_steps)
            self.model.load_state_dict(torch.load(files['pth'], map_location='cpu', weights_only=True))
            self.model.eval()
            self.btn_batch_test.config(state=tk.NORMAL); self.btn_batch_prod.config(state=tk.NORMAL)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {e}")
            self.btn_batch_test.config(state=tk.DISABLED); self.btn_batch_prod.config(state=tk.DISABLED)

    def _create_context_menu(self, tree):
        menu = tk.Menu(self.root, tearoff=0, bg="#313244", fg="#cdd6f4", activebackground="#89b4fa", activeforeground="#1e1e2e", font=("Segoe UI", 10))
        
        def select_all(): tree.selection_set(tree.get_children())
        def copy_selected():
            selected = tree.selection()
            if not selected: return
            text = ""
            for item in selected:
                vals = tree.item(item, 'values')
                text += "\t".join(str(v) for v in vals) + "\n"
            self.root.clipboard_clear(); self.root.clipboard_append(text)

        menu.add_command(label="Select All", command=select_all)
        menu.add_command(label="Copy", command=copy_selected)
        tree.bind("<Button-3>", lambda event: menu.tk_popup(event.x_root, event.y_root))

    def _batch_predict_test(self):
        if self.model is None: return
        file = filedialog.askopenfilename(title="Select Test Data (CSV)", filetypes=[("Data Files", "*.csv *.txt")])
        if not file: return

        batch_win = tk.Toplevel(self.root); batch_win.title("📊 Prediction Results"); batch_win.geometry("900x650"); batch_win.configure(bg="#1e1e2e")
        style = ttk.Style(); style.configure("Treeview", background="#1e1e2e", foreground="#cdd6f4", fieldbackground="#1e1e2e", font=("Segoe UI", 10), rowheight=28); style.configure("Treeview.Heading", background="#313244", foreground="#89b4fa", font=("Segoe UI", 10, "bold")); style.map("Treeview", background=[('selected', '#89b4fa')], foreground=[('selected', '#1e1e2e')])
        
        btn_frame = tk.Frame(batch_win, bg="#1e1e2e"); btn_frame.pack(fill="x", padx=10, pady=(10, 0))
        tk.Button(btn_frame, text="🔍 Explain Selected Row (Grad-CAM)", bg="#cba6f7", fg="#1e1e2e", font=("Segoe UI", 11, "bold"), command=lambda: self._show_gradcam(tree, file)).pack(side="left", padx=5, pady=5)
        
        tree_frame = tk.Frame(batch_win, bg="#1e1e2e"); tree_frame.pack(fill="both", expand=True, padx=10, pady=10)
        tree_y_scroll = ttk.Scrollbar(tree_frame, orient="vertical"); tree_x_scroll = ttk.Scrollbar(tree_frame, orient="horizontal")
        columns = ("Row", "True Label", "Predicted", "Confidence", "Status")
        tree = ttk.Treeview(tree_frame, columns=columns, show="headings", height=20, yscrollcommand=tree_y_scroll.set, xscrollcommand=tree_x_scroll.set)
        tree_y_scroll.config(command=tree.yview); tree_x_scroll.config(command=tree.xview); tree_y_scroll.pack(side="right", fill="y"); tree_x_scroll.pack(side="bottom", fill="x"); tree.pack(side="left", fill="both", expand=True)
        
        for col in columns: tree.heading(col, text=col); tree.column(col, width=150, anchor="center")
        self._create_context_menu(tree)
        
        correct_count = 0; total_count = 0
        with torch.no_grad():
            with open(file, 'r') as f: lines = f.readlines()
            for i, line in enumerate(lines):
                if not line.strip(): continue
                parts = line.strip().split(',')
                if len(parts) < 2: continue
                try: features = [float(x) for x in parts[:-1]]; true_label_str = parts[-1].strip()
                except ValueError: continue 
                
                features_np = np.array(features, dtype=np.float32)
                if self.feat_mean is not None and self.feat_std is not None:
                    _mean = np.array(self.feat_mean, dtype=np.float32); _std = np.array(self.feat_std, dtype=np.float32) + 1e-8
                    features_np = (features_np - _mean) / _std
                tensor_in = torch.tensor(features_np, dtype=torch.float32).unsqueeze(0)
                outputs = self.model(tensor_in); probabilities = torch.nn.functional.softmax(outputs[0], dim=0); confidence, pred_idx = torch.max(probabilities, 0)
                pred_class_name = self.class_names[pred_idx.item()] if pred_idx.item() < len(self.class_names) else str(pred_idx.item())
                status = "❓ Unknown"
                if true_label_str:
                    if str(pred_class_name).upper() == str(true_label_str).upper(): status = "✅ Correct"; correct_count += 1
                    else: status = "❌ Wrong"
                conf_percent = confidence.item() * 100
                tree.insert("", "end", values=(f"Line {i+1}", true_label_str, pred_class_name, f"{conf_percent:.2f}%", status)); total_count += 1

        acc_str = "N/A" if total_count == 0 else f"{(correct_count / total_count) * 100:.2f}%"
        tk.Label(batch_win, text=f"Total: {total_count} | Correct: {correct_count} | Accuracy: {acc_str}", bg="#1e1e2e", fg="#a6e3a1", font=("Segoe UI", 14, "bold")).pack(pady=10)

    def _show_gradcam(self, tree, filepath):
        selected = tree.selection()
        if not selected:
            messagebox.showwarning("Select Row", "Please select a row from the table first."); return
        
        row_idx = int(tree.item(selected[0], 'values')[0].split()[1]) - 1
        true_label_str = tree.item(selected[0], 'values')[1]
        
        with open(filepath, 'r') as f: lines = f.readlines()
        line = lines[row_idx].strip()
        parts = line.split(',')
        features = [float(x) for x in parts[:-1]]
        
        features_np = np.array(features, dtype=np.float32)
        if self.feat_mean is not None and self.feat_std is not None:
            _mean = np.array(self.feat_mean, dtype=np.float32); _std = np.array(self.feat_std, dtype=np.float32) + 1e-8
            features_np = (features_np - _mean) / _std
        
        tensor_in = torch.tensor(features_np, dtype=torch.float32).unsqueeze(0)
        
        cam = GradCAM1D(self.model, self.model.target_layer)
        heatmap, pred_class = cam.generate(tensor_in)
        cam.remove_hooks()

        win = tk.Toplevel(self.root); win.title(f"🔍 Grad-CAM Explanation for Line {row_idx+1}"); win.geometry("900x650"); win.configure(bg="#1e1e2e")
        
        fig = Figure(figsize=(8, 4), facecolor='#1e1e2e')
        ax = fig.add_subplot(111)
        ax.set_facecolor('#181825'); ax.tick_params(colors='#cdd6f4')
        for s in ax.spines.values(): s.set_color('#585b70')
        
        x_axis = np.arange(len(features))
        ax.plot(x_axis, features, color='#89b4fa', alpha=0.8, label='Raw Signal')
        
        heatmap_upsampled = np.interp(np.linspace(0, len(heatmap), len(features)), np.arange(len(heatmap)), heatmap)
        
        extent = [0, len(features), min(features), max(features)]
        ax.imshow(heatmap_upsampled[np.newaxis, :], aspect='auto', cmap='jet', alpha=0.5, extent=extent)
        
        ax.set_title(f"Predicted Class: {self.class_names[pred_class]} | Model Attention (Red = High Focus)", color='#a6e3a1', fontsize=12)
        ax.set_xlabel("Time Step", color='#cdd6f4'); ax.set_ylabel("Signal Value", color='#cdd6f4')
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=win); canvas.get_tk_widget().pack(fill="both", expand=False)
        canvas.draw()

        # 🌟 NEW: LLM AUDITOR UI
        llm_frame = tk.LabelFrame(win, text=" 🧠 LLM Clinical Auditor (Explain AI Failure) ", bg="#1e1e2e", fg="#cba6f7", font=("Segoe UI", 11, "bold"), bd=1, relief="solid")
        llm_frame.pack(fill="x", padx=10, pady=10)
        
        row_api = tk.Frame(llm_frame, bg="#1e1e2e"); row_api.pack(fill="x", pady=5)
        tk.Label(row_api, text="OpenAI API Key:", bg="#1e1e2e", fg="#cdd6f4", font=("Segoe UI", 10)).pack(side="left", padx=5)
        api_entry = tk.Entry(row_api, show="*", width=40, bg="#181825", fg="#a6e3a1", font=("Consolas", 10))
        api_entry.pack(side="left", padx=5)
        
        btn_llm = tk.Button(row_api, text="🧠 Ask LLM Auditor", bg="#cba6f7", fg="#1e1e2e", font=("Segoe UI", 11, "bold"),
                            command=lambda: self._run_llm_audit(features, heatmap_upsampled, true_label_str, self.class_names[pred_class], api_entry.get()))
        btn_llm.pack(side="left", padx=10)

    def _run_llm_audit(self, raw_features, heatmap, true_label, pred_label, api_key):
        if not api_key:
            messagebox.showwarning("API Key Missing", "لطفاً کلید API خود را وارد کنید.")
            return
        
        if openai is None:
            messagebox.showerror("Library Missing", "کتابخانه openai نصب نیست.\nلطفاً در ترمینال اجرا کنید:\npip install openai")
            return

        # ساخت متن ویژگی‌ها
        feature_text = ""
        for name, val in zip(CTG_FEATURE_NAMES, raw_features):
            feature_text += f"- {name}: {val:.2f}\n"
            
        # پیدا کردن ۳ ویژگی با بیشترین توجه
        top_indices = np.argsort(heatmap)[-3:][::-1]
        attention_text = ""
        for idx in top_indices:
            fname = CTG_FEATURE_NAMES[idx] if idx < len(CTG_FEATURE_NAMES) else f"Feature {idx}"
            attention_text += f"- {fname} (Weight: {heatmap[idx]:.2f})\n"

        prompt = f"""
        You are an expert obstetrician and an AI auditor.
        
        Here is the patient's clinical data (21 extracted CTG features):
        {feature_text}
        
        True Clinical Label: {true_label}
        AI (1D-CNN) Model Prediction: {pred_label}
        
        The Explainable AI (Grad-CAM) report shows what the AI model actually focused on:
        {attention_text}
        
        Your Task:
        1. As an obstetrician, what is the true medical diagnosis based on the clinical data?
        2. Did the AI model look at the right features? Explain why the AI model failed or succeeded based on its attention map.
        3. Provide a concise clinical explanation of the AI's failure mode (e.g., Sparse Feature Attention).
        """

        loading_win = tk.Toplevel(self.root); loading_win.title("Analyzing..."); loading_win.geometry("300x100"); loading_win.configure(bg="#1e1e2e")
        tk.Label(loading_win, text="🧠 LLM is analyzing the clinical data...\nPlease wait.", bg="#1e1e2e", fg="#a6e3a1", font=("Segoe UI", 11)).pack(expand=True)

        def fetch_llm():
            try:
                client = openai.OpenAI(api_key=api_key)
                response = client.chat.completions.create(
                    model="gpt-4o-mini", # یا gpt-4o برای دقت بالاتر
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2
                )
                result_text = response.choices[0].message.content
                
                self.root.after(0, lambda: self._show_llm_result(result_text, loading_win))
            except Exception as e:
                self.root.after(0, lambda: [loading_win.destroy(), messagebox.showerror("LLM Error", str(e))])

        threading.Thread(target=fetch_llm, daemon=True).start()

    def _show_llm_result(self, result_text, loading_win):
        loading_win.destroy()
        res_win = tk.Toplevel(self.root)
        res_win.title("🧠 LLM Clinical Audit Result")
        res_win.geometry("700x600")
        res_win.configure(bg="#1e1e2e")
        
        txt = tk.Text(res_win, wrap="word", bg="#181825", fg="#a6e3a1", font=("Segoe UI", 12), padx=10, pady=10)
        txt.insert("1.0", result_text)
        txt.config(state="disabled")
        txt.pack(fill="both", expand=True, padx=10, pady=10)

    def _batch_predict_production(self):
        if self.model is None: messagebox.showwarning("No Model", "Please load a model first."); return
        use_cnn_stem = True if "CIFAR" in self.dataset_name else False
        
        if use_cnn_stem:
            folder = filedialog.askdirectory(title="Select Folder with CIFAR Images")
            if not folder: return
            files_to_process = glob.glob(os.path.join(folder, "*.png")) + glob.glob(os.path.join(folder, "*.jpg"))
        else:
            file = filedialog.askopenfilename(title="Select File with Multiple Patterns (CSV)", filetypes=[("Data Files", "*.csv *.txt")])
            if not file: return
            files_to_process = [file]

        if not files_to_process: messagebox.showwarning("Empty", "No files found to process."); return

        batch_win = tk.Toplevel(self.root); batch_win.title("🏭 Production Results (Unlabeled Data)"); batch_win.geometry("600x600"); batch_win.configure(bg="#1e1e2e")
        style = ttk.Style(); style.configure("Treeview", background="#1e1e2e", foreground="#cdd6f4", fieldbackground="#1e1e2e", font=("Segoe UI", 10), rowheight=28); style.configure("Treeview.Heading", background="#313244", foreground="#89b4fa", font=("Segoe UI", 10, "bold")); style.map("Treeview", background=[('selected', '#89b4fa')], foreground=[('selected', '#1e1e2e')])
        
        tree_frame = tk.Frame(batch_win, bg="#1e1e2e"); tree_frame.pack(fill="both", expand=True, padx=10, pady=10)
        tree_y_scroll = ttk.Scrollbar(tree_frame, orient="vertical"); tree_x_scroll = ttk.Scrollbar(tree_frame, orient="horizontal")
        
        columns = ("Source", "Predicted")
        tree = ttk.Treeview(tree_frame, columns=columns, show="headings", height=20, yscrollcommand=tree_y_scroll.set, xscrollcommand=tree_x_scroll.set)
        tree_y_scroll.config(command=tree.yview); tree_x_scroll.config(command=tree.xview); tree_y_scroll.pack(side="right", fill="y"); tree_x_scroll.pack(side="bottom", fill="x"); tree.pack(side="left", fill="both", expand=True)
        
        for col, txt in zip(columns, ["File/Row", "Predicted"]):
            tree.heading(col, text=txt); tree.column(col, width=250, anchor="center")
        self._create_context_menu(tree)

        total_count = 0
        transform_img = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        with torch.no_grad():
            for filepath in files_to_process:
                filename = os.path.basename(filepath)
                if use_cnn_stem:
                    img = Image.open(filepath).convert('RGB'); tensor_in = transform_img(img).unsqueeze(0)
                    outputs = self.model(tensor_in); probabilities = torch.nn.functional.softmax(outputs[0], dim=0); _, pred_idx = torch.max(probabilities, 0)
                    tree.insert("", "end", values=(filename, self.class_names[pred_idx.item()])); total_count += 1
                else:
                    try:
                        with open(filepath, 'r') as f: lines = f.readlines()
                        for i, line in enumerate(lines):
                            if not line.strip(): continue
                            parts = line.strip().split(',')
                            try: features = [float(x) for x in parts]
                            except ValueError: continue 
                            
                            features_np = np.array(features, dtype=np.float32)
                            if self.feat_mean is not None and self.feat_std is not None:
                                _mean = np.array(self.feat_mean, dtype=np.float32); _std = np.array(self.feat_std, dtype=np.float32) + 1e-8
                                features_np = (features_np - _mean) / _std
                            else:
                                if features_np.max() > 1.0: features_np = features_np / 255.0
                                features_np = features_np * 2.0 - 1.0
                            
                            tensor_in = torch.tensor(features_np, dtype=torch.float32).unsqueeze(0)
                            outputs = self.model(tensor_in); probabilities = torch.nn.functional.softmax(outputs[0], dim=0); _, pred_idx = torch.max(probabilities, 0)
                            pred_class_name = self.class_names[pred_idx.item()] if pred_idx.item() < len(self.class_names) else str(pred_idx.item())
                            tree.insert("", "end", values=(f"Line {i+1}", pred_class_name)); total_count += 1
                    except Exception as e: print(f"Error reading {filename}: {e}")

        tk.Label(batch_win, text=f"Total Patterns: {total_count}", bg="#1e1e2e", fg="#89b4fa", font=("Segoe UI", 14, "bold")).pack(pady=10)


# ============================================================
# 🖥 رابط گرافیکی اصلی
# ============================================================
class App:
    def __init__(self, root):
        self.root = root; root.title(f"Multi-Dataset Discovery (Bipolar Input + AdamW) | Workers={MAX_WORKERS}")
        try: root.state('zoomed')
        except tk.TclError: root.geometry('1400x900')
        root.configure(bg="#1e1e2e")
        
        self.custom_datasets = {}; self.pop_boost = 0; self.output_dir = RESULTS_DIR
        self.q = queue.Queue(); self.stop_event = threading.Event(); self.pause_event = threading.Event()
        self.thread = None; self.start_time = None; self.heartbeat = ["⏳", "⌛"]; self.hb_idx = 0
        self._build(); self._poll(); self._animate_heartbeat(); self._update_timer()

    def _clear_output_dir(self):
        try:
            if os.path.exists(self.output_dir):
                for item in os.listdir(self.output_dir):
                    item_path = os.path.join(self.output_dir, item)
                    try:
                        if os.path.isdir(item_path): shutil.rmtree(item_path)
                        else: os.remove(item_path)
                    except Exception as e: print(f"⚠️ حذف {item_path} ناموفق: {e}")
        except Exception as e: print(f"⚠️ پاک‌سازی پوشه results ناموفق: {e}")

    def _build(self):
        st = ttk.Style(); st.theme_use('clam')
        st.configure(".", background="#1e1e2e", foreground="#cdd6f4"); st.configure("TLabel", background="#1e1e2e", foreground="#cdd6f4"); st.configure("TFrame", background="#1e1e2e")
        st.configure("TLabelframe", background="#1e1e2e", foreground="#89b4fa", font=("Segoe UI", 10, "bold"), borderwidth=1, relief="solid"); st.configure("TLabelframe.Label", background="#1e1e2e", foreground="#89b4fa")
        st.configure("TNotebook", background="#1e1e2e", borderwidth=0, tabmargins=[10, 5, 10, 0])
        st.configure("TNotebook.Tab", background="#313244", foreground="#a6adc8", font=("Segoe UI", 12, "bold"), padding=[25, 10], borderwidth=0, lightcolor="#313244", darkcolor="#313244", bordercolor="#313244")
        st.map("TNotebook.Tab", background=[("selected", "#cba6f7"), ("active", "#585b70")], foreground=[("selected", "#1e1e2e"), ("active", "#cdd6f4")], lightcolor=[("selected", "#cba6f7")], darkcolor=[("selected", "#cba6f7")], bordercolor=[("selected", "#cba6f7")], expand=[("selected", [1, 1, 1, 0])])
        st.configure("Vertical.TScrollbar", background="#313244", troughcolor="#181825", bordercolor="#1e1e2e", arrowcolor="#cdd6f4"); st.configure("Horizontal.TScrollbar", background="#313244", troughcolor="#181825", bordercolor="#1e1e2e", arrowcolor="#cdd6f4")
        
        self.notebook = ttk.Notebook(self.root); self.notebook.pack(fill="both", expand=True, padx=5, pady=5)
        self.tab_editor = ttk.Frame(self.notebook, style="TFrame"); self.notebook.add(self.tab_editor, text=" 📝 Input Train Data ")
        self.tab_train = ttk.Frame(self.notebook, style="TFrame"); self.notebook.add(self.tab_train, text=" 🏋️ Training & Visualization ")
        self.tab_predict = ttk.Frame(self.notebook, style="TFrame"); self.notebook.add(self.tab_predict, text=" 🔍 Recognition Test Data ")

        contact_frame = tk.Frame(self.tab_editor, bg="#181825", bd=1, relief=tk.SOLID); contact_frame.pack(side="top", fill="x", padx=10, pady=(10, 5), ipady=8)
        contact_frame.columnconfigure(0, weight=1); contact_frame.columnconfigure(1, weight=1); contact_frame.columnconfigure(2, weight=1)
        tk.Label(contact_frame, text="👤 Ahmad Mehrju", bg="#181825", fg="#89b4fa", font=("Segoe UI", 14, "bold")).grid(row=0, column=0, sticky="w", padx=20)
        tk.Label(contact_frame, text="✉  mehrjoot42@gmail.com", bg="#181825", fg="#a6e3a1", font=("Segoe UI", 12, "bold")).grid(row=0, column=1, sticky="n")
        tk.Label(contact_frame, text="📞  09197526216", bg="#181825", fg="#f9e2af", font=("Segoe UI", 12, "bold")).grid(row=0, column=2, sticky="e", padx=20)

        note_frame = ttk.LabelFrame(self.tab_editor, text=" 📝 Input Train Data (راست‌کلیک: Select All / Copy / Paste / Cut / Clear) ", padding=5); note_frame.pack(side="top", fill="both", expand=True, padx=10, pady=(5, 10))
        note_toolbar = tk.Frame(note_frame, bg="#1e1e2e"); note_toolbar.pack(fill="x", pady=(0, 5))
        
        tk.Button(note_toolbar, text="💾 SAVE as Train.csv", bg="#a6e3a1", fg="#1e1e2e", font=("Segoe UI", 11, "bold"), relief=tk.FLAT, command=self._save_train_csv).pack(side="left", padx=5, pady=5)
        tk.Button(note_toolbar, text="💾 SAVE as Test.csv", bg="#f9e2af", fg="#1e1e2e", font=("Segoe UI", 11, "bold"), relief=tk.FLAT, command=self._save_test_csv).pack(side="left", padx=5, pady=5)
        tk.Button(note_toolbar, text="💾 SAVE as Production", bg="#cba6f7", fg="#1e1e2e", font=("Segoe UI", 11, "bold"), relief=tk.FLAT, command=self._save_production_csv).pack(side="left", padx=5, pady=5)

        editor_inner = tk.Frame(note_frame, bg="#ffffff"); editor_inner.pack(fill="both", expand=True)
        self.note_text = tk.Text(editor_inner, wrap="none", height=20, bg="#ffffff", fg="#888888", font=("Consolas", 12), undo=True, insertbackground="#000000", selectbackground="#b3d4fc", selectforeground="#000000")

        self.placeholder_text = (
            "0,1,1,1,0,1,0,0,0,1,1,1,1,1,1,1,0,0,0,1,1,0,0,0,1,A\n" "0,1,1,1,0,1,0,0,0,1,1,1,1,1,1,1,0,0,0,1,1,0,0,0,1,A\n" "0,1,1,1,0,1,0,0,0,1,1,1,1,1,1,1,0,0,0,1,1,0,0,0,1,A\n"
            "1,1,1,1,0,1,0,0,0,1,1,1,1,1,0,1,0,0,0,1,1,1,1,1,0,B\n" "1,1,1,1,0,1,0,0,0,1,1,1,1,1,0,1,0,0,0,1,1,1,1,1,0,B\n" "1,1,1,1,0,1,0,0,0,1,1,1,1,1,0,1,0,0,0,1,1,1,1,1,0,B\n"
            "0,1,1,1,0,1,0,0,0,1,1,0,0,0,0,1,0,0,0,1,0,1,1,1,0,C\n" "0,1,1,1,0,1,0,0,0,1,1,0,0,0,0,1,0,0,0,1,0,1,1,1,0,C\n" "0,1,1,1,0,1,0,0,0,1,1,0,0,0,0,1,0,0,0,1,0,1,1,1,0,C"
        )
        
        self.production_placeholder_text = (
            "0,1,1,1,0,1,0,0,0,1,1,1,1,1,1,1,0,0,0,1,1,0,0,0,1\n" "0,1,1,1,0,1,0,0,0,1,1,1,1,1,1,1,0,0,0,1,1,0,0,0,1\n" "0,1,1,1,0,1,0,0,0,1,1,1,1,1,1,1,0,0,0,1,1,0,0,0,1\n"
            "1,1,1,1,0,1,0,0,0,1,1,1,1,1,0,1,0,0,0,1,1,1,1,1,0\n" "1,1,1,1,0,1,0,0,0,1,1,1,1,1,0,1,0,0,0,1,1,1,1,1,0\n" "1,1,1,1,0,1,0,0,0,1,1,1,1,1,0,1,0,0,0,1,1,1,1,1,0\n"
            "0,1,1,1,0,1,0,0,0,1,1,0,0,0,0,1,0,0,0,1,0,1,1,1,0\n" "0,1,1,1,0,1,0,0,0,1,1,0,0,0,0,1,0,0,0,1,0,1,1,1,0\n" "0,1,1,1,0,1,0,0,0,1,1,0,0,0,0,1,0,0,0,1,0,1,1,1,0"
        )
        
        self.note_text.insert("1.0", self.placeholder_text)
        self.note_text.bind("<FocusIn>", self._on_text_focus_in); self.note_text.bind("<FocusOut>", self._on_text_focus_out)

        note_sb_y = ttk.Scrollbar(editor_inner, orient="vertical", command=self.note_text.yview); note_sb_x = ttk.Scrollbar(editor_inner, orient="horizontal", command=self.note_text.xview)
        self.note_text.config(yscrollcommand=note_sb_y.set, xscrollcommand=note_sb_x.set)
        self.note_text.grid(row=0, column=0, sticky="nsew"); note_sb_y.grid(row=0, column=1, sticky="ns"); note_sb_x.grid(row=1, column=0, sticky="ew")
        editor_inner.rowconfigure(0, weight=1); editor_inner.columnconfigure(0, weight=1)

        self.note_menu = tk.Menu(self.root, tearoff=0, bg="#313244", fg="#cdd6f4", activebackground="#89b4fa", activeforeground="#1e1e2e", font=("Segoe UI", 10))
        self.note_menu.add_command(label="✂️  Cut", command=self._note_cut); self.note_menu.add_command(label="📋  Copy", command=self._note_copy); self.note_menu.add_command(label="📥  Paste", command=self._note_paste)
        self.note_menu.add_separator(); self.note_menu.add_command(label="🔤  Select All", command=self._note_select_all); self.note_menu.add_separator(); self.note_menu.add_command(label="🗑️  Clear All", command=self._note_clear)
        
        self.note_text.bind("<Button-3>", self._show_note_menu); self.note_text.bind("<Button-2>", self._show_note_menu)
        self.note_text.bind("<Control-a>", lambda e: self._note_select_all()); self.note_text.bind("<Control-A>", lambda e: self._note_select_all())
        self.note_text.bind("<Control-v>", self._note_paste); self.note_text.bind("<Control-V>", self._note_paste); self.note_text.bind("<<Paste>>", self._note_paste)

        cf = ttk.LabelFrame(self.tab_train, text="⚙ Configuration", padding=10); cf.pack(fill="x", padx=10, pady=5)
        row_top = ttk.Frame(cf); row_top.pack(fill="x", pady=8)
        for i in range(4): row_top.columnconfigure(i, weight=1, uniform="group1")
        
        box_bg = "#313244"; box_fg = "#cdd6f4"; font_bold = ("Segoe UI", 10, "bold"); font_normal = ("Segoe UI", 9)
        
        ds_frame = tk.Frame(row_top, bg=box_bg, bd=1, relief=tk.SOLID, highlightbackground="#89b4fa", highlightthickness=1); ds_frame.grid(row=0, column=0, padx=5, sticky="nsew")
        tk.Label(ds_frame, text="Dataset", bg=box_bg, fg="#89b4fa", font=font_bold).pack(pady=(8, 2))
        self.current_dataset_lbl = tk.Label(ds_frame, text="None Selected", bg=box_bg, fg="#f9e2af", font=("Segoe UI", 10, "bold")); self.current_dataset_lbl.pack(pady=2)
        ds_inner = tk.Frame(ds_frame, bg=box_bg); ds_inner.pack(pady=(5, 10))
        self.dataset_var = tk.StringVar(value="")
        tk.Button(ds_inner, text="➕ ADD CSV DATA", bg="#cba6f7", fg="#1e1e2e", font=("Segoe UI", 12, "bold"), command=self._add_custom_dataset, padx=20, pady=5).pack()

        info_frame = tk.Frame(row_top, bg=box_bg, bd=1, relief=tk.SOLID, highlightbackground="#89b4fa", highlightthickness=1); info_frame.grid(row=0, column=1, padx=5, sticky="nsew")
        tk.Label(info_frame, text="📊 Dataset Info", bg=box_bg, fg="#89b4fa", font=font_bold).pack(pady=(8, 2))
        self.dataset_info_lbl = tk.Label(info_frame, text="", bg=box_bg, fg=box_fg, font=font_normal); self.dataset_info_lbl.pack(pady=(0, 8))

        evo_frame = tk.Frame(row_top, bg=box_bg, bd=1, relief=tk.SOLID, highlightbackground="#89b4fa", highlightthickness=1); evo_frame.grid(row=0, column=2, padx=5, sticky="nsew")
        tk.Label(evo_frame, text="🧬 Evolution", bg=box_bg, fg="#89b4fa", font=font_bold).pack(pady=(8, 2))
        evo_inner = tk.Frame(evo_frame, bg=box_bg); evo_inner.pack(pady=(0, 8))
        tk.Label(evo_inner, text="GEN 1:", bg=box_bg, fg=box_fg, font=font_normal).pack(side="left")
        self.v_gen1 = tk.StringVar(value="5"); ttk.Entry(evo_inner, textvariable=self.v_gen1, width=4, justify="center", font=font_normal).pack(side="left", padx=(2, 10))
        tk.Label(evo_inner, text="GEN 2:", bg=box_bg, fg=box_fg, font=font_normal).pack(side="left")
        self.v_gen2 = tk.StringVar(value="5"); ttk.Entry(evo_inner, textvariable=self.v_gen2, width=4, justify="center", font=font_normal).pack(side="left", padx=2)

        train_frame = tk.Frame(row_top, bg=box_bg, bd=1, relief=tk.SOLID, highlightbackground="#a6e3a1", highlightthickness=1); train_frame.grid(row=0, column=3, padx=5, sticky="nsew")
        tk.Label(train_frame, text="🏋️ Training Batches", bg=box_bg, fg="#a6e3a1", font=font_bold).pack(pady=(8, 2))
        train_inner = tk.Frame(train_frame, bg=box_bg); train_inner.pack(pady=(0, 5))
        tk.Label(train_inner, text="Phase 1:", bg=box_bg, fg=box_fg, font=font_normal).pack(side="left")
        self.v_batches1 = tk.StringVar(value="1000"); self.batches1_combo = ttk.Combobox(train_inner, textvariable=self.v_batches1, state="normal", width=5, font=font_normal); self.batches1_combo['values'] = [str(x) for x in range(50, 2100, 50)]; self.batches1_combo.pack(side="left", padx=(2, 10))
        tk.Label(train_inner, text="Phase 2:", bg=box_bg, fg=box_fg, font=font_normal).pack(side="left")
        self.v_batches2 = tk.StringVar(value="1000"); self.batches2_combo = ttk.Combobox(train_inner, textvariable=self.v_batches2, state="normal", width=5, font=font_normal); self.batches2_combo['values'] = [str(x) for x in range(50, 2100, 50)]; self.batches2_combo.pack(side="left", padx=2)
        self.auto_batches_var = tk.BooleanVar(value=True); tk.Checkbutton(train_frame, text="AutoML Scaling", variable=self.auto_batches_var, bg=box_bg, fg=box_fg, font=font_normal, selectcolor=box_bg, activebackground=box_bg, activeforeground=box_fg).pack(pady=(0, 8))

        self._update_dataset_info()

        row2 = ttk.Frame(cf); row2.pack(fill="x", pady=(10, 0))
        self.hb_lbl = tk.Label(row2, text="⏸", bg="#1e1e2e", fg="#f38ba8", font=("Segoe UI", 28, "bold")); self.hb_lbl.pack(side="left", padx=(10, 20))
        self.btn_start = tk.Button(row2, text="▶ START", command=self._start, bg="#a6e3a1", fg="#1e1e2e", font=("Segoe UI", 12, "bold"), padx=20, pady=5); self.btn_start.pack(side="left", padx=10)
        self.btn_pause = tk.Button(row2, text="⏸ PAUSE", command=self._toggle_pause, bg="#89b4fa", fg="#1e1e2e", font=("Segoe UI", 12, "bold"), padx=20, pady=5, state=tk.DISABLED); self.btn_pause.pack(side="left", padx=10)
        self.btn_stop = tk.Button(row2, text="⏹ STOP", command=self._stop, bg="#f38ba8", fg="#1e1e2e", font=("Segoe UI", 12, "bold"), padx=20, pady=5, state=tk.DISABLED); self.btn_stop.pack(side="left", padx=10)
        self.btn_more_train = tk.Button(row2, text="🚀 More Train", command=self._more_train, bg="#cba6f7", fg="#1e1e2e", font=("Segoe UI", 12, "bold"), padx=20, pady=5); self.btn_more_train.pack(side="left", padx=10)

        pf = ttk.LabelFrame(self.tab_train, text="⏱ Status & Progress", padding=10); pf.pack(fill="x", padx=10, pady=5)
        row_t = ttk.Frame(pf); row_t.pack(fill="x", pady=(0, 5))
        out_dir_name = os.path.basename(self.output_dir); self.dir_lbl = tk.StringVar(value=f"Output Folder: {out_dir_name}"); tk.Label(row_t, textvariable=self.dir_lbl, bg="#1e1e2e", fg="#a6e3a1", font=("Consolas", 11, "bold")).pack(side="left", padx=(0, 20))
        self.time_lbl = tk.StringVar(value="Elapsed: 00:00:00"); tk.Label(row_t, textvariable=self.time_lbl, bg="#1e1e2e", fg="#f9e2af", font=("Consolas", 11)).pack(side="right")
        self.stage_lbl = tk.StringVar(value="Ready"); tk.Label(row_t, textvariable=self.stage_lbl, bg="#1e1e2e", fg="#89b4fa", font=("Segoe UI", 11, "bold")).pack(side="left")

        r1 = ttk.Frame(pf); r1.pack(fill="x", pady=2)
        tk.Label(r1, text="Phase 1:", bg="#1e1e2e", fg="#a6e3a1", font=("Segoe UI", 10, "bold")).pack(side="left", padx=(0, 10)); self.pv1 = tk.DoubleVar(value=0); ttk.Progressbar(r1, variable=self.pv1, maximum=100).pack(side="left", fill="x", expand=True)
        r2 = ttk.Frame(pf); r2.pack(fill="x", pady=2)
        tk.Label(r2, text="Phase 2:", bg="#1e1e2e", fg="#89b4fa", font=("Segoe UI", 10, "bold")).pack(side="left", padx=(0, 10)); self.pv2 = tk.DoubleVar(value=0); ttk.Progressbar(r2, variable=self.pv2, maximum=100).pack(side="left", fill="x", expand=True)

        mf = tk.PanedWindow(self.tab_train, orient=tk.HORIZONTAL, bg="#1e1e2e", bd=0, sashwidth=8); mf.pack(fill="both", expand=True, padx=10, pady=5)
        df = ttk.Frame(mf); self.fig = Figure(figsize=(7, 4.5), facecolor='#1e1e2e')
        self.ax1 = self.fig.add_subplot(2, 1, 1); self._style(self.ax1, 'Gen', 'Best Acc (%)', 'Phase 1 (Arch)'); self.ax2 = self.fig.add_subplot(2, 1, 2); self._style(self.ax2, 'Gen', 'Best Acc (%)', 'Phase 2 (Adam coeffs)')
        self.fig.tight_layout(); self.canvas = FigureCanvasTkAgg(self.fig, master=df); self.canvas.get_tk_widget().pack(fill="both", expand=True)
        lf = ttk.LabelFrame(mf, text="📋 Live Log", padding=5); self.log = tk.Text(lf, wrap="word", bg="#181825", fg="#cdd6f4", font=("Consolas", 11)); self.log.pack(fill="both", expand=True)
        sb = ttk.Scrollbar(lf, command=self.log.yview); sb.pack(side="right", fill="y"); self.log.config(yscrollcommand=sb.set)
        self.log.tag_config("header", foreground="#89b4fa", font=("Consolas", 11, "bold")); self.log.tag_config("success", foreground="#a6e3a1"); self.log.tag_config("error", foreground="#f38ba8")
        mf.add(df, stretch="always", minsize=300); mf.add(lf, stretch="always", minsize=300)

        self.predictor = PredictorApp(self.tab_predict)

    def _on_text_focus_in(self, event):
        if self.note_text.get("1.0", "end-1c") in [self.placeholder_text, self.production_placeholder_text]:
            self.note_text.delete("1.0", tk.END); self.note_text.config(fg="#000000")

    def _on_text_focus_out(self, event):
        if not self.note_text.get("1.0", "end-1c").strip():
            self.note_text.delete("1.0", tk.END); self.note_text.insert("1.0", self.placeholder_text); self.note_text.config(fg="#888888")

    def _save_train_csv(self):
        content = self.note_text.get("1.0", "end-1c").strip()
        if not content or content in [self.placeholder_text.strip(), self.production_placeholder_text.strip()]:
            messagebox.showwarning("Empty Data", "لطفاً ابتدا داده‌های آموزشی را وارد کنید."); return
        filepath = os.path.join(SCRIPT_DIR, "Train.csv")
        try:
            with open(filepath, "w", encoding="utf-8") as f: f.write(content)
            messagebox.showinfo("Success", f"داده‌ها با موفقیت در فایل ذخیره شدند:\n{filepath}")
        except Exception as e: messagebox.showerror("Error", f"خطا در ذخیره فایل:\n{e}")

    def _save_test_csv(self):
        content = self.note_text.get("1.0", "end-1c").strip()
        if not content or content in [self.placeholder_text.strip(), self.production_placeholder_text.strip()]:
            messagebox.showwarning("Empty Data", "لطفاً ابتدا داده‌های تست را وارد کنید."); return
        filepath = os.path.join(SCRIPT_DIR, "Test.csv")
        try:
            with open(filepath, "w", encoding="utf-8") as f: f.write(content)
            messagebox.showinfo("Success", f"داده‌ها با موفقیت در فایل ذخیره شدند:\n{filepath}")
        except Exception as e: messagebox.showerror("Error", f"خطا در ذخیره فایل:\n{e}")

    def _save_production_csv(self):
        content = self.note_text.get("1.0", "end-1c").strip()
        if not content or content in [self.placeholder_text.strip(), self.production_placeholder_text.strip()]:
            self.note_text.delete("1.0", tk.END); self.note_text.insert("1.0", self.production_placeholder_text); self.note_text.config(fg="#888888"); return
        filepath = os.path.join(SCRIPT_DIR, "Production.csv")
        try:
            with open(filepath, "w", encoding="utf-8") as f: f.write(content)
            messagebox.showinfo("Success", f"داده‌ها با موفقیت در فایل ذخیره شدند:\n{filepath}")
        except Exception as e: messagebox.showerror("Error", f"خطا در ذخیره فایل:\n{e}")

    def _show_note_menu(self, event):
        try: self.note_menu.tk_popup(event.x_root, event.y_root)
        finally: self.note_menu.grab_release()

    def _note_select_all(self):
        self.note_text.tag_add("sel", "1.0", "end-1c"); self.note_text.mark_set("insert", "1.0"); self.note_text.see("insert"); return "break"

    def _note_copy(self):
        try:
            sel = self.note_text.get("sel.first", "sel.last"); self.root.clipboard_clear(); self.root.clipboard_append(sel)
        except tk.TclError: pass
        return "break"

    def _note_paste(self, event=None):
        try:
            text = self.root.clipboard_get()
            if self.note_text.get("1.0", "end-1c") in [self.placeholder_text, self.production_placeholder_text]:
                self.note_text.delete("1.0", tk.END); self.note_text.config(fg="#000000")
            try: self.note_text.delete("sel.first", "sel.last")
            except tk.TclError: pass
            self.note_text.insert("insert", text)
        except tk.TclError: pass
        return "break"

    def _note_cut(self):
        try:
            sel = self.note_text.get("sel.first", "sel.last"); self.root.clipboard_clear(); self.root.clipboard_append(sel); self.note_text.delete("sel.first", "sel.last")
        except tk.TclError: pass
        return "break"

    def _note_clear(self):
        if messagebox.askyesno("Clear Notes", "پاک کردن تمام متن دفترچه؟"): self.note_text.delete("1.0", tk.END)

    def _add_custom_dataset(self):
        filepath = filedialog.askopenfilename(title="Select CSV Dataset", filetypes=[("CSV Files", "*.csv")])
        if not filepath: return
        filename = os.path.basename(filepath); name = f"Custom: {filename}"
        self.custom_datasets[name] = {"path": filepath}
        self.dataset_var.set(name); self.current_dataset_lbl.config(text=filename)
        self._update_dataset_info(); messagebox.showinfo("Success", f"دیتاست '{filename}' با موفقیت اضافه شد!")

    def _more_train(self):
        if self.thread and self.thread.is_alive():
            messagebox.showwarning("در حال اجرا", "لطفاً ابتدا فرآیند آموزش را متوقف کنید (STOP) و صبر کنید تا متوقف شود."); return
        try:
            current_b1 = int(self.v_batches1.get()); current_b2 = int(self.v_batches2.get()); current_gen1 = int(self.v_gen1.get()); current_gen2 = int(self.v_gen2.get())
        except ValueError: messagebox.showerror("خطا", "مقادیر Batches یا GEN نامعتبر هستند."); return

        new_b1 = current_b1 * 2; new_b2 = current_b2 * 2; new_gen1 = current_gen1 + 1; new_gen2 = current_gen2 + 1
        self.v_batches1.set(str(new_b1)); self.v_batches2.set(str(new_b2)); self.v_gen1.set(str(new_gen1)); self.v_gen2.set(str(new_gen2))
        self.pop_boost += 1; self.auto_batches_var.set(False)
        
        self.log.insert("end", f"🚀 More Train: Batches → P1={new_b1}, P2={new_b2} | GEN → {new_gen1}/{new_gen2} | Genomes +{self.pop_boost}\n", "success"); self.log.see("end")
        messagebox.showinfo("More Train Activated 🚀", f"✅ Batches دو برابر شدند:\n   Phase 1: {current_b1} → {new_b1}\n   Phase 2: {current_b2} → {new_b2}\n\n✅ نسل‌ها (GEN) +1:\n   GEN 1: {current_gen1} → {new_gen1}\n   GEN 2: {current_gen2} → {new_gen2}\n\n✅ ژنوم‌ها: +{self.pop_boost} در هر دو فاز\n\n⚠️ AutoML Scaling خاموش شد.\n\n🔄 هر دو فاز از ابتدا شروع می‌شوند...")
        self.root.after(50, self._start)

    def _update_dataset_info(self, event=None):
        name = self.dataset_var.get()
        if not name: self.dataset_info_lbl.config(text="Please ADD a CSV Dataset"); return
        if name in self.custom_datasets: self.dataset_info_lbl.config(text="Custom CSV Loaded\nAutoML Enabled"); return
        configs = load_configs()
        if name in configs:
            cfg = configs[name]; total = cfg['n_train'] + cfg['n_val'] + cfg['n_test']
            self.dataset_info_lbl.config(text=f"Total: {total:,} | Train: {cfg['n_train']:,} | Val: {cfg['n_val']:,} | Test: {cfg['n_test']:,}")
        else: self.dataset_info_lbl.config(text="")

    def _style(self, ax, xl, yl, t):
        ax.set_facecolor('#181825'); ax.tick_params(colors='#cdd6f4', labelsize=9)
        for s in ax.spines.values(): s.set_color('#585b70')
        ax.grid(True, alpha=0.2); ax.set_xlabel(xl, color='#cdd6f4', fontsize=9); ax.set_ylabel(yl, color='#cdd6f4', fontsize=9); ax.set_title(t, color='#89b4fa', fontsize=10, fontweight='bold')

    def _animate_heartbeat(self):
        if self.thread and self.thread.is_alive():
            if self.pause_event.is_set(): self.hb_lbl.config(text="⏸", fg="#f9e2af")
            else: self.hb_idx ^= 1; self.hb_lbl.config(text=self.heartbeat[self.hb_idx], fg="#a6e3a1")
        else: self.hb_lbl.config(text="⏸", fg="#f38ba8")
        self.root.after(500, self._animate_heartbeat)

    def _update_timer(self):
        if (self.start_time and self.thread and self.thread.is_alive() and not self.pause_event.is_set()):
            el = time.time() - self.start_time; m, s = divmod(int(el), 60); h, m = divmod(m, 60); self.time_lbl.set(f"Elapsed: {h:02d}:{m:02d}:{s:02d}")
        self.root.after(1000, self._update_timer)

    def _start(self):
        if self.thread and self.thread.is_alive(): return
        try:
            gen1 = int(self.v_gen1.get()); gen2 = int(self.v_gen2.get()); manual_override = not self.auto_batches_var.get(); batches1 = int(self.v_batches1.get()); batches2 = int(self.v_batches2.get())
        except ValueError: self.stage_lbl.set("❌ مقادیر عددی نامعتبر"); return
        
        dataset = self.dataset_var.get()
        if not dataset: self.stage_lbl.set("❌ لطفاً ابتدا یک دیتاست CSV اضافه کنید"); return

        gui_cfg = {'datasets': [dataset], 'num_seeds': 1, 'gen1': gen1, 'gen2': gen2, 'feedback_steps': 1, 'batches1': batches1, 'batches2': batches2, 'manual_override': manual_override, 'custom_datasets': self.custom_datasets, 'pop_boost': self.pop_boost}

        self.ax1.clear(); self._style(self.ax1, 'Gen', 'Best Acc (%)', 'Phase 1 (Arch)'); self.ax2.clear(); self._style(self.ax2, 'Gen', 'Best Acc (%)', 'Phase 2 (Adam coeffs)')
        self.canvas.draw_idle(); self.pv1.set(0); self.pv2.set(0); self.stage_lbl.set("Initializing..."); self.start_time = time.time(); self.stop_event.clear(); self.pause_event.clear(); self.log.delete("1.0", tk.END)
        
        self._clear_output_dir()
        self.log.insert("end", "🗑️ پوشه results پاک‌سازی شد.\n", "info"); self.dir_lbl.set(f"Output Folder: {os.path.basename(self.output_dir)}")

        self.btn_start.config(state=tk.DISABLED); self.btn_pause.config(state=tk.NORMAL, text="⏸ PAUSE", bg="#89b4fa"); self.btn_stop.config(state=tk.NORMAL)

        self.thread = threading.Thread(target=run_experiment, args=(gui_cfg, self.q, self.stop_event, self.pause_event, self.output_dir), daemon=True); self.thread.start()

    def _toggle_pause(self):
        if self.thread and self.thread.is_alive():
            if not self.pause_event.is_set(): self.pause_event.set(); self.btn_pause.config(text="▶ RESUME", bg="#f9e2af"); self.stage_lbl.set("⏸ Paused...")
            else: self.pause_event.clear(); self.btn_pause.config(text="⏸ PAUSE", bg="#89b4fa"); self.stage_lbl.set("▶ Running...")

    def _stop(self):
        self.stop_event.set(); self.pause_event.clear(); self.btn_stop.config(state=tk.DISABLED); self.btn_pause.config(state=tk.DISABLED, text="⏸ PAUSE", bg="#89b4fa"); self.stage_lbl.set("Stopping...")

    def _show_finish_window(self, stats=None):
        if stats is None: stats = {}
        
        win = tk.Toplevel(self.root); win.title("✅ Scientific Report"); win.configure(bg="#1e1e2e")
        win.update_idletasks(); w, h = 650, 480; sw = win.winfo_screenwidth(); sh = win.winfo_screenheight(); x = (sw - w) // 2; y = (sh - h) // 2; win.geometry(f"{w}x{h}+{x}+{y}"); win.resizable(False, False); win.attributes("-topmost", True)
        
        tk.Label(win, text="✅ Evolution Complete", bg="#1e1e2e", fg="#a6e3a1", font=("Segoe UI", 24, "bold")).pack(pady=(25, 10))
        
        ds_name = stats.get('dataset', 'N/A')
        comp_pct = stats.get('complexity', 0.0)
        lin_acc = stats.get('linear_acc', 0.0)
        acc = stats.get('test_acc', 0.0)
        gain = stats.get('complexity_gain', 0.0)
        params = stats.get('model_params', 0)
        
        info_frame = tk.Frame(win, bg="#181825", bd=1, relief=tk.SOLID)
        info_frame.pack(fill="x", padx=40, pady=10, ipady=10)
        
        tk.Label(info_frame, text=f"Dataset: {ds_name}", bg="#181825", fg="#89b4fa", font=("Segoe UI", 14, "bold")).pack(pady=5)
        tk.Label(info_frame, text=f"📊 Data Nonlinearity Score: {comp_pct:.2f}%", bg="#181825", fg="#f9e2af", font=("Segoe UI", 14, "bold")).pack(pady=5)
        
        tk.Label(win, text="────────────────────────", bg="#1e1e2e", fg="#585b70").pack()
        
        tk.Label(win, text=f"📈 Linear Baseline: {lin_acc:.2f}%", bg="#1e1e2e", fg="#cdd6f4", font=("Segoe UI", 14)).pack(pady=(10, 0))
        tk.Label(win, text=f"🚀 Evolved Model Accuracy: {acc:.2f}%", bg="#1e1e2e", fg="#a6e3a1", font=("Segoe UI", 16, "bold")).pack(pady=2)
        
        # هایلایت کردن سود حاصل از تکامل
        gain_color = "#a6e3a1" if gain > 0 else "#f38ba8"
        tk.Label(win, text=f"📈 Complexity Gain: {gain:+.2f}%", bg="#1e1e2e", fg=gain_color, font=("Segoe UI", 18, "bold")).pack(pady=10)
        
        tk.Label(win, text=f"🧠 Model Parameters: {params:,}", bg="#1e1e2e", fg="#cba6f7", font=("Segoe UI", 11)).pack(pady=(0, 15))
        
        tk.Button(win, text="OK", command=win.destroy, bg="#89b4fa", fg="#1e1e2e", font=("Segoe UI", 12, "bold"), padx=40, pady=6).pack(pady=(0, 20))
        win.after(100, lambda: win.attributes("-topmost", True))

    def _poll(self):
        try:
            while True:
                kind, d = self.q.get_nowait()
                if kind == 'log': self.log.insert("end", d['msg'] + "\n", d.get('level', 'info')); self.log.see("end")
                elif kind == 'status': self.stage_lbl.set(d.get('text', ''))
                elif kind == 'progress_p1': self.pv1.set(d.get('pct', 0)); self.stage_lbl.set(d.get('text', ''))
                elif kind == 'progress_p2': self.pv2.set(d.get('pct', 0)); self.stage_lbl.set(d.get('text', ''))
                elif kind == 'update_batches': self.v_batches1.set(str(d['b1'])); self.v_batches2.set(str(d['b2']))
                elif kind == 'reset_plots':
                    self.ax1.clear(); self._style(self.ax1, 'Gen', 'Best Acc (%)', f'P1: {d["name"]} (Seed {d["seed"]})')
                    self.ax2.clear(); self._style(self.ax2, 'Gen', 'Best Acc (%)', f'P2: {d["name"]} (Seed {d["seed"]})'); self.canvas.draw_idle()
                elif kind == 'plot_p1':
                    title = self.ax1.get_title(); self.ax1.clear(); self._style(self.ax1, 'Gen', 'Best Acc (%)', title)
                    self.ax1.plot(range(1, len(d['history']) + 1), d['history'], 'o-', color='#a6e3a1'); self.ax1.axhline(y=d['acc'], color='#f9e2af', ls='--', label=f"Best: {d['acc']:.2f}%"); self.ax1.legend(); self.canvas.draw_idle()
                elif kind == 'plot_p2':
                    title = self.ax2.get_title()
                    if " | LR:" in title: title = title.split(" | LR:")[0]
                    new_title = f"{title} | LR: {d.get('lr', 0.001):.6f}"
                    self.ax2.clear(); self._style(self.ax2, 'Gen', 'Best Acc (%)', new_title)
                    self.ax2.plot(range(1, len(d['history']) + 1), d['history'], 'o-', color='#89b4fa'); self.ax2.axhline(y=d['acc'], color='#f9e2af', ls='--', label=f"Best: {d['acc']:.2f}%"); self.ax2.legend(); self.canvas.draw_idle()
                elif kind == 'show_error':
                    self.stop_event.set(); self.pause_event.clear(); self.stage_lbl.set("❌ Failed"); self.btn_start.config(state=tk.NORMAL); self.btn_pause.config(state=tk.DISABLED, text="⏸ PAUSE", bg="#89b4fa"); self.btn_stop.config(state=tk.DISABLED); messagebox.showerror("خطای یادگیری", d['msg'])
                elif kind == 'finished':
                    self.pv1.set(100); self.pv2.set(100); self.stage_lbl.set("✅ Finished!"); self.log.insert("end", f"\n=== {d['msg']} ===\n", "header"); self.btn_start.config(state=tk.NORMAL); self.btn_pause.config(state=tk.DISABLED, text="⏸ PAUSE", bg="#89b4fa"); self.btn_stop.config(state=tk.DISABLED)
                    self._show_finish_window(stats=d.get('stats', None))
        except queue.Empty: pass
        self.root.after(100, self._poll)

if __name__ == '__main__':
    root = tk.Tk()
    root.withdraw()
    
    is_valid, message = check_trial_license()
    
    if is_valid:
        root.deiconify()
        try: root.iconbitmap("Pic.ico")
        except Exception: pass
        app = App(root)
        try:
            import pyi_splash
            pyi_splash.close()
        except ImportError: pass
        root.mainloop()
    else:
        try:
            import pyi_splash
            pyi_splash.close()
        except ImportError: pass
        messagebox.showerror("پایان اعتبار", message)
        root.destroy()
        sys.exit()
