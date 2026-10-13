"""
SignSense ISL — Indian Sign Language IVP prototype
Compatible target: Python 3.12 + MediaPipe 0.10.21.
Implemented: hand landmark extraction, custom static/dynamic dataset capture, Random Forest training,
hold-out evaluation, webcam inference, prediction CSV logs.
Not a universal ISL translator: vocabulary depends on the dataset collected/imported.
"""
import os, csv, time, threading, collections, traceback
from datetime import datetime
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
import numpy as np
import mediapipe as mp

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
MODEL_DIR = os.path.join(ROOT, "models")
LOG_DIR = os.path.join(ROOT, "logs")
STATIC_CSV = os.path.join(DATA_DIR, "static_landmarks.csv")
SEQUENCE_NPZ = os.path.join(DATA_DIR, "dynamic_sequences.npz")
STATIC_MODEL = os.path.join(MODEL_DIR, "static_rf.joblib")
DYNAMIC_MODEL = os.path.join(MODEL_DIR, "dynamic_rf.joblib")
PRED_LOG = os.path.join(LOG_DIR, "predictions.csv")
os.makedirs(DATA_DIR, exist_ok=True); os.makedirs(MODEL_DIR, exist_ok=True); os.makedirs(LOG_DIR, exist_ok=True)
POINTS = 21
FEATURES = POINTS * 3
SEQ_LEN = 30

def hand_features(hand_landmarks):
    """Return 63 normalized features; wrist origin + scale normalization."""
    pts = np.array([[p.x, p.y, p.z] for p in hand_landmarks.landmark], dtype=np.float32)
    pts -= pts[0]
    scale = float(np.max(np.linalg.norm(pts[:, :2], axis=1)))
    if scale > 1e-6: pts /= scale
    return pts.flatten()

def get_hand_result(hands, frame):
    rgb = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
    return hands.process(rgb)

class SignSense:
    def __init__(self, root):
        self.root=root
        root.title("SignSense ISL | Advanced IVP Lab")
        root.geometry("1240x800"); root.minsize(1050,680); root.configure(bg="#101820")
        self.cap=None; self.hands=None; self.running=False; self.features=None
        self.frame_count=0; self.sequence_buffer=collections.deque(maxlen=SEQ_LEN)
        self.static_model=None; self.dynamic_model=None
        self.last_log=0; self.last_prediction="—"; self.batch_job=None
        self.dynamic_recording=False; self.dynamic_frames=[]; self.dynamic_label=None
        self._style(); self._ui(); self.load_models(); self.refresh_stats()
        root.protocol("WM_DELETE_WINDOW", self.close)

    def _style(self):
        s=ttk.Style(); s.theme_use("clam")
        s.configure("TFrame",background="#101820"); s.configure("Panel.TFrame",background="#172631")
        s.configure("TLabel",background="#101820",foreground="#eaf1f5",font=("Segoe UI",10))
        s.configure("Panel.TLabel",background="#172631",foreground="#eaf1f5")
        s.configure("Title.TLabel",background="#101820",foreground="#f4f7f9",font=("Segoe UI",24,"bold"))
        s.configure("Muted.TLabel",background="#101820",foreground="#9cb0bd")
        s.configure("Metric.TLabel",background="#172631",foreground="#55d6be",font=("Segoe UI",17,"bold"))
        s.configure("TButton",padding=8,font=("Segoe UI",10,"bold"))
        s.configure("Accent.TButton",background="#55d6be",foreground="#102027")
        s.map("Accent.TButton",background=[("active","#83e9d8")])
        s.configure("TNotebook",background="#101820",borderwidth=0)
        s.configure("TNotebook.Tab",padding=(16,10),background="#263945",foreground="#eaf1f5")
        s.map("TNotebook.Tab",background=[("selected","#55d6be")],foreground=[("selected","#102027")])

    def _ui(self):
        top=ttk.Frame(self.root); top.pack(fill="x",padx=22,pady=(16,8))
        ttk.Label(top,text="SignSense ISL",style="Title.TLabel").pack(side="left")
        ttk.Label(top,text="REAL-TIME SIGN RECOGNITION • IVP PROJECT",style="Muted.TLabel").pack(side="left",padx=18,pady=(10,0))
        self.status=tk.StringVar(value="Ready • start camera to capture or recognize signs")
        ttk.Label(self.root,textvariable=self.status,style="Muted.TLabel").pack(anchor="w",padx=25,pady=(0,10))
        m=ttk.Frame(self.root); m.pack(fill="x",padx=22,pady=(0,12))
        self.m_samples=self.metric(m,"STATIC SAMPLES","0"); self.m_classes=self.metric(m,"STATIC CLASSES","0")
        self.m_model=self.metric(m,"MODELS","Not trained"); self.m_pred=self.metric(m,"LIVE OUTPUT","—")
        self.tabs=ttk.Notebook(self.root); self.tabs.pack(fill="both",expand=True,padx=22,pady=(0,18))
        self.live=ttk.Frame(self.tabs); self.capture=ttk.Frame(self.tabs); self.train=ttk.Frame(self.tabs); self.info=ttk.Frame(self.tabs)
        for tab,title in [(self.live," Live Detection "),(self.capture," Dataset Studio "),(self.train," Train & Evaluate "),(self.info," Scope & Guide ")]:
            self.tabs.add(tab,text=title)
        self._live_ui(); self._capture_ui(); self._train_ui(); self._info_ui()

    def metric(self,parent,title,value):
        f=ttk.Frame(parent,style="Panel.TFrame",padding=(14,11)); f.pack(side="left",fill="x",expand=True,padx=(0,9))
        ttk.Label(f,text=title,style="Panel.TLabel").pack(anchor="w")
        v=ttk.Label(f,text=value,style="Metric.TLabel"); v.pack(anchor="w",pady=(6,0)); return v

    def _live_ui(self):
        outer=ttk.Frame(self.live,padding=14); outer.pack(fill="both",expand=True)
        panel=ttk.Frame(outer,style="Panel.TFrame",padding=10); panel.pack(side="left",fill="both",expand=True,padx=(0,12))
        self.video=tk.Label(panel,text="WEBCAM PREVIEW\n\nClick Start Camera",bg="#0b1117",fg="#91a7b5",font=("Segoe UI",16))
        self.video.pack(fill="both",expand=True)
        side=ttk.Frame(outer,style="Panel.TFrame",padding=16,width=255); side.pack(side="right",fill="y"); side.pack_propagate(False)
        ttk.Label(side,text="LIVE RECOGNITION",style="Panel.TLabel").pack(anchor="w")
        self.pred=tk.Label(side,text="—",bg="#172631",fg="#55d6be",font=("Segoe UI",34,"bold")); self.pred.pack(anchor="w",pady=(12,4))
        self.conf=tk.StringVar(value="Confidence: —"); ttk.Label(side,textvariable=self.conf,style="Panel.TLabel").pack(anchor="w")
        self.mode=tk.StringVar(value="Mode: static + dynamic"); ttk.Label(side,textvariable=self.mode,style="Panel.TLabel").pack(anchor="w",pady=(5,12))
        ttk.Button(side,text="Start Camera",style="Accent.TButton",command=self.start_camera).pack(fill="x",pady=4)
        ttk.Button(side,text="Stop Camera",command=self.stop_camera).pack(fill="x",pady=4)
        ttk.Button(side,text="Export Prediction Log",command=self.export_log).pack(fill="x",pady=4)
        ttk.Separator(side).pack(fill="x",pady=14)
        ttk.Label(side,text="Recognition notes",style="Panel.TLabel").pack(anchor="w")
        ttk.Label(side,text="• Static model predicts individual poses\n• Dynamic model predicts a 30-frame sequence\n• Train both using your own labeled examples\n• Predictions are limited to trained labels",style="Panel.TLabel",wraplength=215,justify="left").pack(anchor="w",pady=7)

    def _capture_ui(self):
        f=ttk.Frame(self.capture,padding=22); f.pack(fill="both",expand=True)
        ttk.Label(f,text="Dataset Studio",style="Title.TLabel").pack(anchor="w",pady=(0,8))
        ttk.Label(f,text="Build a labeled dataset with your webcam. Use short, consistent labels such as HELLO, YES, NO, THANKS or alphabet labels.",style="Muted.TLabel",wraplength=1000).pack(anchor="w",pady=(0,18))
        row=ttk.Frame(f); row.pack(fill="x",pady=5)
        ttk.Label(row,text="Sign label:").pack(side="left")
        self.label=tk.StringVar(value="HELLO"); ttk.Entry(row,textvariable=self.label,width=22).pack(side="left",padx=10)
        ttk.Button(row,text="Capture 1 static sample",style="Accent.TButton",command=self.capture_static).pack(side="left",padx=4)
        ttk.Button(row,text="Capture 30 static samples",command=lambda:self.capture_static_batch(30)).pack(side="left",padx=4)
        ttk.Button(row,text="Capture 80 static samples",command=lambda:self.capture_static_batch(80)).pack(side="left",padx=4)
        ttk.Separator(f).pack(fill="x",pady=18)
        ttk.Label(f,text="Dynamic gesture capture (movement over time)",font=("Segoe UI",13,"bold")).pack(anchor="w",pady=(0,8))
        ttk.Label(f,text="Start camera first. Enter a label, click Record 2-second gesture, then perform the gesture continuously until the countdown finishes. Repeat each gesture 20–50 times with natural variation.",wraplength=1000).pack(anchor="w",pady=(0,10))
        row2=ttk.Frame(f); row2.pack(fill="x")
        ttk.Button(row2,text="Record 2-second gesture",style="Accent.TButton",command=self.record_dynamic).pack(side="left")
        self.capture_note=tk.StringVar(value="Camera must be running to capture landmark samples.")
        ttk.Label(f,textvariable=self.capture_note,style="Muted.TLabel",wraplength=950).pack(anchor="w",pady=18)
        ttk.Label(f,text="Data files: data/static_landmarks.csv and data/dynamic_sequences.npz",style="Muted.TLabel").pack(anchor="w")

    def _train_ui(self):
        f=ttk.Frame(self.train,padding=22); f.pack(fill="both",expand=True)
        ttk.Label(f,text="Train & Evaluate",style="Title.TLabel").pack(anchor="w",pady=(0,8))
        ttk.Label(f,text="Static and dynamic models are trained separately. Evaluation uses a stratified hold-out split where possible.",style="Muted.TLabel").pack(anchor="w",pady=(0,18))
        row=ttk.Frame(f); row.pack(fill="x")
        ttk.Button(row,text="Train static model",style="Accent.TButton",command=self.train_static).pack(side="left",padx=(0,8))
        ttk.Button(row,text="Train dynamic model",command=self.train_dynamic).pack(side="left",padx=8)
        ttk.Button(row,text="Train both",command=self.train_both).pack(side="left",padx=8)
        ttk.Button(row,text="Open data folder",command=lambda:os.startfile(DATA_DIR)).pack(side="left",padx=8)
        self.eval_text=tk.Text(f,height=22,bg="#0b1117",fg="#eaf1f5",insertbackground="white",font=("Consolas",10),relief="flat",wrap="word")
        self.eval_text.pack(fill="both",expand=True,pady=(18,0))
        self.eval_text.insert("end","Training report will appear here. Capture examples for at least two labels before training.\n")

    def _info_ui(self):
        f=ttk.Frame(self.info,padding=22); f.pack(fill="both",expand=True)
        ttk.Label(f,text="Implemented vs. data-dependent",style="Title.TLabel").pack(anchor="w",pady=(0,12))
        msg=("IMPLEMENTED IN THIS PROJECT\n"
             "• Webcam frame acquisition and BGR→RGB conversion\n"
             "• MediaPipe Hands landmark detection (21 points / 63 features)\n"
             "• Wrist-origin and scale normalization\n"
             "• Custom static sample capture and dynamic 30-frame sequence capture\n"
             "• Random Forest training, hold-out accuracy, precision/recall/F1 report\n"
             "• Static and dynamic live predictions; CSV prediction logs\n"
             "• Dashboard, project report and viva questions\n\n"
             "REQUIRES DATA / FURTHER DEVELOPMENT\n"
             "• Broad ISL vocabulary: collect/import verified, labeled examples for each sign\n"
             "• Dynamic signs: capture repeated sequences for each movement sign\n"
             "• Full sentence translation: needs temporal models plus grammar/context/language processing\n"
             "• Robust generalization: test with separate signers and sessions, not just adjacent frames\n"
             "• Public-dataset import adapters depend on each dataset's license, labels and file format\n\n"
             "This is an ISL-focused research/learning prototype, not a certified translator. ISL and ASL are different languages; use ISL-specific labels and datasets.")
        ttk.Label(f,text=msg,justify="left",wraplength=1000).pack(anchor="w")

    def start_camera(self):
        if self.running:return
        try:
            self.cap=cv2.VideoCapture(0,cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.cap.release(); self.cap=cv2.VideoCapture(0)
            if not self.cap.isOpened(): raise RuntimeError("Webcam could not be opened. Close other camera apps or try camera index 1.")
            self.hands=mp.solutions.hands.Hands(static_image_mode=False,max_num_hands=1,model_complexity=1,
                       min_detection_confidence=.55,min_tracking_confidence=.5)
            self.running=True; self.status.set("Camera running • show one hand clearly"); self._tick()
        except Exception as e:
            messagebox.showerror("Camera startup error",str(e)); self.stop_camera()

    def _tick(self):
        if not self.running or self.cap is None:return
        ok,frame=self.cap.read()
        if ok:
            frame=cv2.flip(frame,1); rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            res=self.hands.process(rgb); feat=None
            if res.multi_hand_landmarks:
                lm=res.multi_hand_landmarks[0]; feat=hand_features(lm); self.features=feat
                mp.solutions.drawing_utils.draw_landmarks(frame,lm,mp.solutions.hands.HAND_CONNECTIONS)
            else:self.features=None
            label="—"; confidence=None; kind="static"
            if feat is not None and self.static_model is not None:
                try:
                    label=str(self.static_model.predict([feat])[0])
                    if hasattr(self.static_model,"predict_proba"): confidence=float(max(self.static_model.predict_proba([feat])[0]))
                except Exception: pass
            self.sequence_buffer.append(feat if feat is not None else np.zeros(FEATURES,dtype=np.float32))
            if self.dynamic_model is not None and len(self.sequence_buffer)==SEQ_LEN and feat is not None:
                try:
                    seq=np.array(self.sequence_buffer,dtype=np.float32).flatten()
                    dp=str(self.dynamic_model.predict([seq])[0])
                    if dp and dp!="": label=dp; kind="dynamic"
                    if hasattr(self.dynamic_model,"predict_proba"): confidence=float(max(self.dynamic_model.predict_proba([seq])[0]))
                except Exception: pass
            if label!="—":
                self.last_prediction=label; self.pred.config(text=label)
                self.conf.set(f"Confidence: {confidence*100:.1f}%" if confidence is not None else "Confidence: —")
                self.mode.set(f"Mode: {kind}")
                cv2.putText(frame,f"{label}  {confidence*100:.0f}%" if confidence is not None else label,(18,42),cv2.FONT_HERSHEY_SIMPLEX,.95,(70,220,190),2,cv2.LINE_AA)
                self.log_prediction(label,confidence,kind)
            else:
                self.pred.config(text="—"); self.conf.set("Confidence: —")
                cv2.putText(frame,"Show a hand / train a model",(18,42),cv2.FONT_HERSHEY_SIMPLEX,.7,(220,220,220),2,cv2.LINE_AA)
            from PIL import Image,ImageTk
            img=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)); img.thumbnail((850,600))
            imgtk=ImageTk.PhotoImage(img); self.video.imgtk=imgtk; self.video.configure(image=imgtk,text="")
            self.refresh_stats()
        self.root.after(20,self._tick)

    def _label_text(self):
        x=self.label.get().strip().upper()
        if not x: messagebox.showwarning("Label required","Enter a sign label first."); return None
        return x[:40]

    def capture_static(self):
        label=self._label_text()
        if not label:return
        if self.features is None: self.capture_note.set("No hand detected. Start camera and place a hand in frame."); return
        self._save_static(label,self.features); self.capture_note.set(f"Saved static sample for {label}."); self.refresh_stats()

    def capture_static_batch(self,n):
        label=self._label_text()
        if not label:return
        if not self.running: messagebox.showwarning("Camera stopped","Start camera first.");return
        self.batch_label=label; self.batch_remaining=n; self.batch_saved=0
        self.capture_note.set(f"Capturing {n} samples for {label}. Hold sign and vary position slightly.")
        self._batch_tick()

    def _batch_tick(self):
        if self.batch_remaining<=0:
            self.capture_note.set(f"Batch complete: saved {self.batch_saved} samples for {self.batch_label}.");self.refresh_stats();return
        if self.features is not None:
            self._save_static(self.batch_label,self.features);self.batch_remaining-=1;self.batch_saved+=1
        self.root.after(100,self._batch_tick)

    def _save_static(self,label,feat):
        new=not os.path.exists(STATIC_CSV)
        with open(STATIC_CSV,"a",newline="",encoding="utf-8") as f:
            w=csv.writer(f)
            if new:w.writerow(["label"]+[f"f{i}" for i in range(FEATURES)]+["timestamp"])
            w.writerow([label]+[float(x) for x in feat]+[datetime.now().isoformat(timespec="seconds")])

    def record_dynamic(self):
        label=self._label_text()
        if not label:return
        if not self.running:messagebox.showwarning("Camera stopped","Start camera first.");return
        if self.dynamic_recording:return
        self.dynamic_recording=True;self.dynamic_label=label;self.dynamic_frames=[]
        self.capture_note.set("Recording gesture for 2 seconds — perform the full movement now.")
        self._dynamic_tick(time.time()+2.0)

    def _dynamic_tick(self,end_time):
        if not self.dynamic_recording:return
        if self.features is not None:self.dynamic_frames.append(self.features.copy())
        if time.time()<end_time:
            self.root.after(30,lambda:self._dynamic_tick(end_time));return
        self.dynamic_recording=False
        if len(self.dynamic_frames)<5:
            self.capture_note.set("Too few hand frames captured. Try again with hand visible throughout.");return
        arr=np.array(self.dynamic_frames,dtype=np.float32)
        # Resample sequence to a fixed length for the model.
        indices=np.linspace(0,len(arr)-1,SEQ_LEN).astype(int); seq=arr[indices]
        if os.path.exists(SEQUENCE_NPZ):
            old=np.load(SEQUENCE_NPZ,allow_pickle=True)
            X=list(old["X"]); y=list(old["y"])
        else:X=[];y=[]
        X.append(seq);y.append(self.dynamic_label)
        np.savez_compressed(SEQUENCE_NPZ,X=np.array(X,dtype=np.float32),y=np.array(y,dtype=str))
        self.capture_note.set(f"Saved dynamic sequence for {self.dynamic_label} ({len(self.dynamic_frames)} frames). Repeat 20–50 times per gesture.")
        self.refresh_stats()

    def _split(self,X,y):
        from sklearn.model_selection import train_test_split
        counts=np.unique(y,return_counts=True)[1]
        strat=y if len(counts)>1 and counts.min()>=2 else None
        return train_test_split(X,y,test_size=.2,random_state=42,stratify=strat)

    def train_static(self):
        try:
            import pandas as pd, joblib
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
            if not os.path.exists(STATIC_CSV):raise ValueError("No static dataset. Capture samples first.")
            df=pd.read_csv(STATIC_CSV); counts=df.label.value_counts()
            if df.label.nunique()<2 or counts.min()<2:raise ValueError("Need at least two labels and at least two samples per label.")
            X=df[[f"f{i}" for i in range(FEATURES)]].to_numpy(dtype=np.float32);y=df.label.astype(str).to_numpy()
            Xtr,Xte,ytr,yte=self._split(X,y)
            clf=RandomForestClassifier(n_estimators=300,class_weight="balanced",random_state=42,n_jobs=-1)
            clf.fit(Xtr,ytr); pred=clf.predict(Xte); acc=accuracy_score(yte,pred)
            joblib.dump(clf,STATIC_MODEL);self.static_model=clf
            report=f"STATIC MODEL EVALUATION\nSamples: {len(y)} | Classes: {len(np.unique(y))}\nHold-out accuracy: {acc:.4f} ({acc*100:.2f}%)\n\n{classification_report(yte,pred,zero_division=0)}\nConfusion matrix (true rows / predicted columns; class order {list(clf.classes_)}):\n{confusion_matrix(yte,pred,labels=clf.classes_)}\n\nNote: adjacent webcam frames may be highly similar; use separate sessions/signers for a stronger evaluation."
            self._show_report(report);self.status.set("Static model trained");self.refresh_stats()
        except Exception as e:messagebox.showerror("Static training failed",str(e))

    def train_dynamic(self):
        try:
            import joblib
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
            if not os.path.exists(SEQUENCE_NPZ):raise ValueError("No dynamic sequences. Record examples first.")
            d=np.load(SEQUENCE_NPZ,allow_pickle=True); X=d["X"].reshape(len(d["X"]),-1); y=d["y"].astype(str)
            classes,counts=np.unique(y,return_counts=True)
            if len(classes)<2 or counts.min()<2:raise ValueError("Need at least two dynamic labels and two sequences per label; 20–50 per label is recommended.")
            Xtr,Xte,ytr,yte=self._split(X,y)
            clf=RandomForestClassifier(n_estimators=300,class_weight="balanced",random_state=42,n_jobs=-1)
            clf.fit(Xtr,ytr);pred=clf.predict(Xte);acc=accuracy_score(yte,pred)
            joblib.dump(clf,DYNAMIC_MODEL);self.dynamic_model=clf
            report=f"DYNAMIC MODEL EVALUATION\nSequences: {len(y)} | Classes: {len(classes)}\nHold-out accuracy: {acc:.4f} ({acc*100:.2f}%)\n\n{classification_report(yte,pred,zero_division=0)}\nConfusion matrix (class order {list(clf.classes_)}):\n{confusion_matrix(yte,pred,labels=clf.classes_)}"
            self._show_report(report);self.status.set("Dynamic model trained");self.refresh_stats()
        except Exception as e:messagebox.showerror("Dynamic training failed",str(e))

    def train_both(self):
        self.train_static()
        if os.path.exists(SEQUENCE_NPZ):self.train_dynamic()

    def _show_report(self,text):
        self.eval_text.delete("1.0","end");self.eval_text.insert("end",text)

    def load_models(self):
        try:
            import joblib
            if os.path.exists(STATIC_MODEL):self.static_model=joblib.load(STATIC_MODEL)
            if os.path.exists(DYNAMIC_MODEL):self.dynamic_model=joblib.load(DYNAMIC_MODEL)
        except Exception as e:self.status.set(f"Model load warning: {e}")

    def refresh_stats(self):
        try:
            rows=0; labels=set()
            if os.path.exists(STATIC_CSV):
                with open(STATIC_CSV,encoding="utf-8") as f:
                    r=csv.DictReader(f)
                    for row in r:rows+=1;labels.add(row["label"])
            self.m_samples.config(text=str(rows));self.m_classes.config(text=str(len(labels)))
            models=("Static " if self.static_model is not None else "")+("Dynamic" if self.dynamic_model is not None else "")
            self.m_model.config(text=models or "Not trained");self.m_pred.config(text=self.last_prediction)
        except Exception:pass

    def log_prediction(self,label,confidence,kind):
        now=time.time()
        if now-self.last_log<.8:return
        self.last_log=now;new=not os.path.exists(PRED_LOG)
        with open(PRED_LOG,"a",newline="",encoding="utf-8") as f:
            w=csv.writer(f)
            if new:w.writerow(["timestamp","label","confidence","mode"])
            w.writerow([datetime.now().isoformat(timespec="seconds"),label,"" if confidence is None else round(confidence,4),kind])

    def export_log(self):
        if not os.path.exists(PRED_LOG):messagebox.showinfo("No log","Run detection first.");return
        target=filedialog.asksaveasfilename(defaultextension=".csv",initialfile="isl_predictions.csv",filetypes=[("CSV","*.csv")])
        if target:
            import shutil;shutil.copy2(PRED_LOG,target);messagebox.showinfo("Exported",target)

    def stop_camera(self):
        self.running=False
        if self.cap is not None:
            try:self.cap.release()
            except Exception:pass
            self.cap=None
        if self.hands is not None:
            try:self.hands.close()
            except Exception:pass
            self.hands=None
        self.video.configure(image="",text="WEBCAM PREVIEW\n\nClick Start Camera");self.video.imgtk=None
        self.status.set("Camera stopped")

    def close(self):
        self.stop_camera();self.root.destroy()

if __name__=="__main__":
    root=tk.Tk(); app=SignSense(root); root.mainloop()
