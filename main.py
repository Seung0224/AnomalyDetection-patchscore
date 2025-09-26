# main.py
import os
import time
import customtkinter as ctk
from tkinter import filedialog, messagebox
from tkinter import font as tkfont
from PIL import Image
import threading

from RunSequence import anomaly_detect
from model_loader import (
    load_model, validate_model_dir,            # 기존 부트용(기본 폴더 자동 로드)
    validate_model_files, load_model_from_files  # ⬅ 새로 추가: 파일로 직접 로드
)

# 기존 Patchcore 사용 시 이미지 전체 학습을 진행하기때문에 학습 HyperParameter를 아무리 강화해도 해결되지않는 문제점이있음.
# Patchsize, stride, augment, etc
# ROI를 지정하여 Patchcore가 학습하는 이미지 영역을 제한함으로써, 불필요한 패치들을 제거하고, 정상 패치들로만 학습이 가능하도록 개선.

# Inference 속도 향상을 위해 OpenVINO 사용 옵션 추가 (OpenVINO가 설치되어있고, CPU가 Intel일때만 사용 권장)
# 백본 (특징 추출기) 모델이 OpenVINO로 변환되어 사용됨. (pytorch → onnx → openvino)
# 기존대비 100~200ms 향상됨.

# 성능을 더끌어올리기위해선 train이미지의 ROI 영역을 1:1로 매칭하고
# 추론할때도 train이미지처럼 ROI 영역을 지정해주는것이 좋음.


# PowerSheell에서 실행 시:
#$env:USE_OPENVINO = "0"
#python .\main.py

#$env:USE_OPENVINO = "1"
#python .\main.py
# 첫 1회는 컴파일/캐시로 느릴 수 있음 → 버리고

DEFAULT_MODEL_DIR = r"D:\ADI\patchcore\models\Bottle"
PATCHCORE_DIR     = r"D:\ADI\patchcore"

# -----------------------------
# Style / Layout Constants
# -----------------------------
SIDE_PAD_X = 16
SIDE_PAD_Y = 16

FRAME1_COLOR = "#BFD5F0"
FRAME2_COLOR = "#89B3F7"
FRAME3_COLOR = "#809CF7"

BTN_FG        = "#D3D3D3"
BTN_TX        = "black"
BTN_HOVER     = "#E5E7EB"
BTN_FONT_SIZE = 16

CHK_TX        = "black"
CHK_HOVER     = "#BFDBFE"
CHK_FONT_SIZE = 14

LBL_TX        = "black"
LBL_FONT_SIZE = 14

F1_MARGIN_X = 10
F1_MARGIN_Y = 10
FRAME2_Y    = 12
FRAME3_Y    = 12
F2_MARGIN_X = 10
F3_MARGIN_X = 10

BTN_PAD_Y      = 6
NAV_GAP_X      = 5
SECTION_GAP_Y  = 6

PATH_CTRL_H    = 36

PREFERRED_FONTS = ["Segoe UI", "Noto Sans KR", "Malgun Gothic", "Arial", "Helvetica", "sans-serif"]

class App(ctk.CTk):
    def __init__(self):
        # --- fields ---
        self._src = None
        self._view = None
        self._ctk_img = None
        self.image_list = []
        self.image_index = -1
        self.current_image_path = None
        self.model_dir     = DEFAULT_MODEL_DIR
        self.patchcore_dir = PATCHCORE_DIR
        self.model = None
        self.device = None

        # model file selections
        self.faiss_path = None
        self.pkl_path   = None

        self.auto_infer_var = None
        self.save_dir_var = None
        self.save_origin_var = None
        self.save_overlay_var = None
        self.default_save_dir = None

        self._infer_running = False
        self._last_inferred_path = None
        self._last_proc_ms = None

        # --- UI ---
        super().__init__()
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        self.title("ADI Inspector")
        self.after(0, lambda: self.state("zoomed"))

        # fonts
        try:
            fams = set(tkfont.families())
        except Exception:
            fams = set()
        self.ui_font_family = next((f for f in PREFERRED_FONTS if f in fams), PREFERRED_FONTS[0])
        self.btn_font = ctk.CTkFont(family=self.ui_font_family, size=BTN_FONT_SIZE)
        self.lbl_font = ctk.CTkFont(family=self.ui_font_family, size=LBL_FONT_SIZE)
        self.chk_font = ctk.CTkFont(family=self.ui_font_family, size=CHK_FONT_SIZE)

        # default save dir
        self.default_save_dir = self._compute_default_save_dir()

        # Tk variables
        self.auto_infer_var   = ctk.BooleanVar(value=False)
        self.save_dir_var     = ctk.StringVar(value=self.default_save_dir)
        self.save_origin_var  = ctk.BooleanVar(value=False)
        self.save_overlay_var = ctk.BooleanVar(value=False)

        # for file name labels
        self.faiss_name_var = ctk.StringVar(value="faiss name: (none)")
        self.pkl_name_var   = ctk.StringVar(value="pkl name: (none)")

        # Left image area
        self.image_label = ctk.CTkLabel(self, text="", width=820, height=600, corner_radius=6)
        self.image_label.grid(row=0, column=0, padx=16, pady=16, sticky="nsew")
        self.image_label.bind("<Configure>", lambda e: self._render_to_label())

        # Right side container
        side = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent", border_width=0)
        side.grid(row=0, column=1, padx=(0, SIDE_PAD_X), pady=SIDE_PAD_Y, sticky="ns")
        side.grid_columnconfigure(0, weight=1)

        # -------- Frame #1 (Model Open + file labels) --------
        frame1 = ctk.CTkFrame(side, corner_radius=6, fg_color=FRAME1_COLOR)
        frame1.grid(row=0, column=0, padx=F1_MARGIN_X, pady=(F1_MARGIN_Y, FRAME2_Y), sticky="ew")
        frame1.grid_columnconfigure(0, weight=1)

        # file name labels (above button)
        self.lbl_faiss_name = ctk.CTkLabel(frame1, textvariable=self.faiss_name_var, text_color=LBL_TX, font=self.lbl_font)
        self.lbl_faiss_name.grid(row=0, column=0, padx=10, pady=(10, 2), sticky="w")

        self.lbl_pkl_name = ctk.CTkLabel(frame1, textvariable=self.pkl_name_var, text_color=LBL_TX, font=self.lbl_font)
        self.lbl_pkl_name.grid(row=1, column=0, padx=10, pady=(0, 8), sticky="w")

        self.btn_model = ctk.CTkButton(
            frame1, text="Model Open", command=self.on_model_open,
            fg_color=BTN_FG, text_color=BTN_TX, hover_color=BTN_HOVER, font=self.btn_font
        )
        self.btn_model.grid(row=2, column=0, padx=10, pady=(0, 10), sticky="ew")

        # -------- Frame #2 --------
        frame2 = ctk.CTkFrame(side, corner_radius=6, fg_color=FRAME2_COLOR)
        frame2.grid(row=1, column=0, padx=F2_MARGIN_X, pady=(0, FRAME3_Y), sticky="ew")
        frame2.grid_columnconfigure(0, weight=1)

        self.btn_open = ctk.CTkButton(
            frame2, text="Image Open", command=self.open_image,
            state="disabled", fg_color=BTN_FG, text_color=BTN_TX, hover_color=BTN_HOVER, font=self.btn_font
        )
        self.btn_open.grid(row=0, column=0, padx=10, pady=(10, SECTION_GAP_Y), sticky="ew")

        nav_row = ctk.CTkFrame(frame2, fg_color="transparent")
        nav_row.grid(row=1, column=0, padx=10, pady=(0, SECTION_GAP_Y), sticky="ew")
        nav_row.grid_columnconfigure(0, weight=1)
        nav_row.grid_columnconfigure(1, weight=1)

        self.btn_image_prev = ctk.CTkButton(
            nav_row, text="Prev", command=self.prev_image,
            state="disabled", fg_color=BTN_FG, text_color=BTN_TX, hover_color=BTN_HOVER, font=self.btn_font
        )
        self.btn_image_prev.grid(row=0, column=0, padx=(0, NAV_GAP_X), pady=0, sticky="ew")

        self.btn_image_next = ctk.CTkButton(
            nav_row, text="Next", command=self.next_image,
            state="disabled", fg_color=BTN_FG, text_color=BTN_TX, hover_color=BTN_HOVER, font=self.btn_font
        )
        self.btn_image_next.grid(row=0, column=1, padx=(NAV_GAP_X, 0), pady=0, sticky="ew")

        self.btn_infer = ctk.CTkButton(
            frame2, text="Inference", command=self.run_infer,
            state="disabled", fg_color=BTN_FG, text_color=BTN_TX, hover_color=BTN_HOVER, font=self.btn_font
        )
        self.btn_infer.grid(row=2, column=0, padx=10, pady=(0, 6), sticky="ew")

        self.lbl_proc = ctk.CTkLabel(
            frame2, text="Processing Time: N/A", text_color=LBL_TX, font=self.lbl_font,
            anchor="center", justify="center"
        )
        self.lbl_proc.grid(row=3, column=0, padx=10, pady=(0, 10), sticky="ew")

        # -------- Frame #3 --------
        frame3 = ctk.CTkFrame(side, corner_radius=6, fg_color=FRAME3_COLOR)
        frame3.grid(row=2, column=0, padx=F3_MARGIN_X, pady=(0, F1_MARGIN_Y), sticky="ew")
        frame3.grid_columnconfigure(0, weight=1)

        path_row = ctk.CTkFrame(frame3, fg_color="transparent")
        path_row.grid(row=0, column=0, padx=10, pady=(10, 6), sticky="ew")
        path_row.grid_columnconfigure(0, weight=1)
        path_row.grid_columnconfigure(1, weight=0)

        self.entry_save = ctk.CTkEntry(path_row, textvariable=self.save_dir_var, height=PATH_CTRL_H,
                                       fg_color="white", text_color="black", font=self.lbl_font)
        self.entry_save.grid(row=0, column=0, padx=(0, 6), pady=0, sticky="ew")

        self.btn_search = ctk.CTkButton(path_row, text="Search", command=self.on_select_save_dir,
                                        fg_color=BTN_FG, text_color=BTN_TX, hover_color=BTN_HOVER,
                                        height=PATH_CTRL_H, width=90, font=self.btn_font)
        self.btn_search.grid(row=0, column=1, padx=(0, 0), pady=0, sticky="ew")

        opts_row = ctk.CTkFrame(frame3, fg_color="transparent")
        opts_row.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="ew")
        for i in range(3):
            opts_row.grid_columnconfigure(i, weight=1)

        self.chk_auto = ctk.CTkCheckBox(opts_row, text="Auto", font=self.chk_font,
                                        variable=self.auto_infer_var, command=self.on_toggle_auto,
                                        state="disabled", text_color=CHK_TX, hover_color=CHK_HOVER)
        self.chk_auto.grid(row=0, column=0, padx=(0, 6), pady=0, sticky="w")

        self.chk_save_origin = ctk.CTkCheckBox(opts_row, text="Origin", variable=self.save_origin_var,
                                               text_color=CHK_TX, hover_color=CHK_HOVER, font=self.chk_font)
        self.chk_save_origin.grid(row=0, column=1, padx=6, pady=0, sticky="w")

        self.chk_save_overlay = ctk.CTkCheckBox(opts_row, text="Overlay", variable=self.save_overlay_var,
                                                text_color=CHK_TX, hover_color=CHK_HOVER, font=self.chk_font)
        self.chk_save_overlay.grid(row=0, column=2, padx=(6, 0), pady=0, sticky="w")

        # expand grid
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # try boot default model (optional)
        self._try_boot_default_model()
        self._sync_controls()

    # -----------------------------
    # Utils
    # -----------------------------
    def _compute_default_save_dir(self):
        try:
            home = os.path.expanduser("~")
            desktop = os.path.join(home, "Desktop")
            return desktop if os.path.isdir(desktop) else home
        except Exception:
            return os.getcwd()

    def _update_model_file_labels(self):
        faiss_disp = os.path.basename(self.faiss_path) if self.faiss_path else "(none)"
        pkl_disp   = os.path.basename(self.pkl_path) if self.pkl_path else "(none)"
        self.faiss_name_var.set(f"faiss name: {faiss_disp}")
        self.pkl_name_var.set(f"pkl name: {pkl_disp}")

    # -----------------------------
    # Controls state sync / lock
    # -----------------------------
    def _sync_controls(self):
        if self._infer_running:
            return
        self.btn_model.configure(state="normal")

        model_ready = self.model is not None
        img_ready   = self._src is not None
        has_list    = len(self.image_list) > 0

        for w in (self.btn_open, self.chk_auto, self.entry_save, self.btn_search,
                  self.chk_save_origin, self.chk_save_overlay):
            try:
                w.configure(state=("normal" if model_ready else "disabled"))
            except Exception:
                pass

        pn_state = "normal" if has_list else "disabled"
        self.btn_image_prev.configure(state=pn_state)
        self.btn_image_next.configure(state=pn_state)

        inf_state = "normal" if (model_ready and img_ready) else "disabled"
        self.btn_infer.configure(state=inf_state)

    def _lock_controls_for_infer(self):
        for w in (self.btn_model, self.btn_open, self.btn_infer,
                  self.btn_image_prev, self.btn_image_next, self.chk_auto,
                  self.btn_search, self.chk_save_origin, self.chk_save_overlay):
            try: w.configure(state="disabled")
            except Exception: pass
        try: self.entry_save.configure(state="disabled")
        except Exception: pass

    def _unlock_controls_after_infer(self):
        self._sync_controls()

    # -----------------------------
    # Model handling
    # -----------------------------
    def _try_boot_default_model(self):
        ok, msg = validate_model_dir(self.model_dir)
        if not ok:
            self.btn_model.configure(state="normal")
            return
        try:
            self.model, self.device = load_model(self.model_dir, reset_cache=True)
            # default boot: also reflect filenames in labels
            faiss = os.path.join(self.model_dir, "nnscorer_search_index.faiss")
            pkl   = os.path.join(self.model_dir, "patchcore_params.pkl")
            self.faiss_path = faiss if os.path.isfile(faiss) else None
            self.pkl_path   = pkl if os.path.isfile(pkl) else None
            self._update_model_file_labels()
        except Exception as e:
            messagebox.showwarning("Model", f"기본 모델 로드 실패:\n{e}")

    def on_model_open(self):
        if self._infer_running:
            return

        # 1) pick FAISS
        faiss_path = filedialog.askopenfilename(
            title="Select FAISS index file",
            filetypes=[("FAISS index","*.faiss;*.index"), ("All files","*.*")],
            initialdir=r"D:\ADI\patchcore\models"   # 기본 경로 지정
        )
        if not faiss_path:
            return

        # faiss 파일이 선택된 폴더
        faiss_dir = os.path.dirname(faiss_path)

        # 2) pick PKL (같은 폴더에서 시작)
        pkl_path = filedialog.askopenfilename(
            title="Select PatchCore params (PKL/PTH)",
            filetypes=[("Params","*.pkl;*.pth;*.pt"), ("All files","*.*")],
            initialdir=faiss_dir   # ← 이 부분 추가
        )
        if not pkl_path:
            return

        ok, msg = validate_model_files(faiss_path, pkl_path)
        if not ok:
            messagebox.showerror("Model", f"선택한 파일이 올바르지 않습니다.\n{msg}")
            return

        try:
            self.model, self.device = load_model_from_files(faiss_path, pkl_path, reset_cache=True)
            self.model_dir = os.path.dirname(pkl_path)
            self.faiss_path = faiss_path
            self.pkl_path   = pkl_path
            self._update_model_file_labels()
            messagebox.showinfo("Model", "모델 로드 완료")
            self._sync_controls()
            if self.auto_infer_var.get() and self._src is not None:
                self._maybe_start_infer()
        except Exception as e:
            messagebox.showerror("Model", f"모델 로드 실패:\n{e}")

    # -----------------------------
    # Save path selection
    # -----------------------------
    def on_select_save_dir(self):
        if self._infer_running:
            return
        d = filedialog.askdirectory(title="Select folder to save outputs")
        if d:
            self.save_dir_var.set(d)

    # -----------------------------
    # Image navigation
    # -----------------------------
    def open_image(self):
        if self._infer_running:
            return
        path = filedialog.askopenfilename(
            filetypes=[("Images","*.png;*.jpg;*.jpeg;*.bmp")],
            initialdir=r"D:\ADI\dataset"
        )
        self.current_image_path = path
        if not path:
            return

        folder = os.path.dirname(path)
        self.image_list = self._build_image_list(folder)
        try:
            self.image_index = self.image_list.index(path)
        except ValueError:
            fname = os.path.basename(path).lower()
            matches = [i for i, p in enumerate(self.image_list)
                       if os.path.basename(p).lower() == fname]
            self.image_index = matches[0] if matches else 0

        self._load_image_by_index()
        self._sync_controls()
        self._maybe_start_infer()

    def next_image(self):
        if self._infer_running: return
        if not self.image_list: return self.open_image()
        self.image_index = (self.image_index + 1) % len(self.image_list)
        self._load_image_by_index()
        self._sync_controls()
        self._maybe_start_infer()

    def prev_image(self):
        if self._infer_running: return
        if not self.image_list: return self.open_image()
        self.image_index = (self.image_index - 1) % len(self.image_list)
        self._load_image_by_index()
        self._sync_controls()
        self._maybe_start_infer()

    def _load_image_by_index(self):
        try:
            path = self.image_list[self.image_index]
        except (IndexError, ValueError):
            return
        self.current_image_path = path
        self._src = Image.open(path).convert("RGB")
        self._view = self._src.copy()
        self._render_to_label()
        self._set_proc_time(None)

    def _build_image_list(self, folder):
        exts = {".png", ".jpg", ".jpeg", ".bmp"}
        files = [os.path.join(folder, n) for n in os.listdir(folder)
                 if os.path.splitext(n)[1].lower() in exts]
        files.sort(key=lambda p: os.path.basename(p).lower())
        return files

    # -----------------------------
    # Manual / Auto inference
    # -----------------------------
    def run_infer(self):
        if self._infer_running: return
        if self.model is None:
            messagebox.showwarning("ADI", "먼저 모델을 로드하세요.")
            return
        if not getattr(self, "current_image_path", None):
            messagebox.showwarning("ADI", "이미지를 먼저 열어주세요.")
            return
        self._start_infer_thread()

    def on_toggle_auto(self):
        if self._infer_running: return
        if self.auto_infer_var.get() and (self.model is not None) and (self._src is not None):
            self._maybe_start_infer()

    def _maybe_start_infer(self):
        if not self.auto_infer_var.get(): return
        if self.model is None or self._src is None: return
        if self._infer_running: return
        self._start_infer_thread()

    def _start_infer_thread(self):
        image_snapshot = self._src.copy() if self._src is not None else None
        img_path = self.current_image_path
        entered = (self.save_dir_var.get() or "").strip()
        save_dir = entered or self.default_save_dir
        save_origin = bool(self.save_origin_var.get())
        save_overlay = bool(self.save_overlay_var.get())

        self._infer_running = True
        self._lock_controls_for_infer()
        threading.Thread(
            target=self._infer_thread,
            args=(img_path, image_snapshot, save_dir, save_origin, save_overlay),
            daemon=True
        ).start()

    def _infer_thread(self, img_path, image_snapshot, save_dir, save_origin, save_overlay):
        t0 = time.perf_counter()
        try:
            overlay_img, score = anomaly_detect(img=image_snapshot, model=self.model, device=self.device, index=(self.image_index + 1),
                                                total=len(self.image_list), threshold=2.0 )
            print("Anomaly Score:", score)
            err = None
        except Exception as e:
            overlay_img = None
            err = str(e)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        if overlay_img is not None and save_dir and (save_origin or save_overlay):
            try:
                os.makedirs(save_dir, exist_ok=True)
                base = os.path.basename(img_path) if img_path else "image.png"
                stem, ext = os.path.splitext(base)
                if not ext: ext = ".png"
                if save_origin and image_snapshot is not None:
                    image_snapshot.save(os.path.join(save_dir, f"{stem}_org{ext}"))
                if save_overlay:
                    overlay_img.save(os.path.join(save_dir, f"{stem}_overlay{ext}"))
            except Exception:
                pass

        def ui_update():
            try:
                if overlay_img and self.current_image_path == img_path:
                    self._view = overlay_img
                    self._render_to_label()
                    self.update_idletasks()
                    self._set_proc_time(elapsed_ms)
                elif overlay_img is None:
                    self._set_proc_time(None)
                    messagebox.showerror("ADI", f"Overlay 생성 실패\n{err or ''}")
            finally:
                self._infer_running = False
                self._last_inferred_path = img_path
                self._unlock_controls_after_infer()
                if (self.auto_infer_var.get()
                        and self.model is not None
                        and self.current_image_path != self._last_inferred_path):
                    self._maybe_start_infer()

        self.after(0, ui_update)

    def _set_proc_time(self, ms):
        self._last_proc_ms = ms
        if self.lbl_proc is None:
            return
        
        if ms is None:
            msg = f"Processing Time: N/A"
        else:
            msg = f"Processing Time: {ms:.1f} ms"

        self.lbl_proc.configure(text=msg)


    # -----------------------------
    # Rendering
    # -----------------------------
    def _render_to_label(self):
        if self._view is None: return
        w, h = self._view.size
        self._ctk_img = ctk.CTkImage(light_image=self._view, dark_image=self._view, size=(w, h))
        self.image_label.configure(image=self._ctk_img, text="")

if __name__ == "__main__":
    app = App()
    app.mainloop()