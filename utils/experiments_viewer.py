import os
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ==========================================
# CONFIGURACIÓN DE MÉTRICAS Y TIPOS DE GRÁFICO
# ==========================================
METRIC_CONFIG = {
    "train_loss": {"label": "Train Loss", "type": "line"},
    "train_acc": {"label": "Train Accuracy (%)", "type": "line"},
    "test_loss": {"label": "Test Loss", "type": "line"},
    "test_acc": {"label": "Test Accuracy (%)", "type": "line"},
    "epoch_time_sec": {"label": "Epoch Time (sec)", "type": "line"},
    "inference_time_ms_per_sample": {"label": "Inference Time (ms/sample)", "type": "line"},
    "max_vram_mb": {"label": "Max VRAM (MB)", "type": "bar"},
    "model_flops": {"label": "Model FLOPs", "type": "bar"},
    "model_params": {"label": "Model Parameters", "type": "bar"}
}

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

# ==========================================
# CLASE AUXILIAR PARA TOOLTIPS (HOVER)
# ==========================================
class ToolTip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tip_window = None
        self.widget.bind("<Enter>", self.show_tip)
        self.widget.bind("<Leave>", self.hide_tip)

    def show_tip(self, event=None):
        if self.tip_window or not self.text:
            return
        x = self.widget.winfo_rootx() + 25
        y = self.widget.winfo_rooty() + 25
        self.tip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(tw, text=self.text, justify=tk.LEFT,
                         background="#ffffe0", relief=tk.SOLID, borderwidth=1,
                         font=("tahoma", "9", "normal"))
        label.pack(ipadx=1)

    def hide_tip(self, event=None):
        tw = self.tip_window
        self.tip_window = None
        if tw:
            tw.destroy()

# ==========================================
# APLICACIÓN PRINCIPAL
# ==========================================
class AppVisualizador:
    def __init__(self, root):
        self.root = root
        self.root.title("Visualizador de Métricas - TFG")
        self.root.geometry("1050x850") # Un pelín más alta para acomodar la lista vertical
        
        self.loaded_files = {}  # path: {"name": str, "df": DataFrame}
        self.current_metric = "train_loss"
        
        self.setup_ui()

    def setup_ui(self):
        # 1. Zona Superior: Panel de Control de Archivos
        top_frame = tk.Frame(self.root, pady=10, padx=10)
        top_frame.pack(fill=tk.X, side=tk.TOP)
        
        # Subframe izquierdo para el botón de añadir
        btn_frame = tk.Frame(top_frame)
        btn_frame.pack(side=tk.LEFT, anchor="n", padx=(0, 10))
        
        btn_add = tk.Button(btn_frame, text="➕ Añadir\nCSV", command=self.load_file, 
                            bg="#4CAF50", fg="white", font=("Arial", 10, "bold"), 
                            width=10, height=2)
        btn_add.pack()
        
        # Contenedor con Scrollbar VERTICAL para las tarjetas de archivos
        scroll_canvas = tk.Canvas(top_frame, height=100, highlightthickness=1, relief=tk.SUNKEN)
        scrollbar = tk.Scrollbar(top_frame, orient="vertical", command=scroll_canvas.yview)
        
        # El truco para que ocupen todo el ancho es configurar correctamente el canvas y el frame interno
        self.files_container = tk.Frame(scroll_canvas)
        
        # Vincular el scroll dinámico
        self.files_container.bind(
            "<Configure>",
            lambda e: scroll_canvas.configure(scrollregion=scroll_canvas.bbox("all"))
        )
        canvas_window = scroll_canvas.create_window((0, 0), window=self.files_container, anchor="nw")
        
        # Hacer que el frame interno se estire al ancho del canvas
        scroll_canvas.bind('<Configure>', lambda event: scroll_canvas.itemconfigure(canvas_window, width=event.width))
        
        scroll_canvas.configure(yscrollcommand=scrollbar.set)
        
        scroll_canvas.pack(side=tk.LEFT, fill=tk.X, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # 2. Zona Central: Gráfica Matplotlib
        self.chart_frame = tk.Frame(self.root, bg="white", relief=tk.RIDGE, bd=2)
        self.chart_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.fig, self.ax = plt.subplots(figsize=(6, 4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.chart_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 3. Zona Inferior: Selectores de Métrica (Botones)
        bottom_frame = tk.Frame(self.root, pady=10)
        bottom_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        for i, (metric_id, config) in enumerate(METRIC_CONFIG.items()):
            btn = tk.Button(
                bottom_frame, 
                text=config["label"], 
                command=lambda m=metric_id: self.change_metric(m),
                font=("Arial", 9),
                relief=tk.RAISED,
                bd=2
            )
            row = i // 5
            col = i % 5
            btn.grid(row=row, column=col, padx=5, pady=5, sticky="nsew")
            bottom_frame.grid_columnconfigure(col, weight=1)

        self.update_plot()

    def load_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not file_path or file_path in self.loaded_files:
            return
        
        try:
            df = pd.read_csv(file_path)
            if 'model_flops' in df.columns:
                df['model_flops'] = df['model_flops'].astype(str).str.replace(' MFLOPs', '', case=False).str.strip().astype(float)
                
            filename = os.path.basename(file_path)
            self.loaded_files[file_path] = {"name": filename, "df": df}
            
            self.refresh_file_tags()
            self.update_plot()
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo parsear el archivo:\n{e}")

    def remove_file(self, file_path):
        if file_path in self.loaded_files:
            del self.loaded_files[file_path]
            self.refresh_file_tags()
            self.update_plot()

    def refresh_file_tags(self):
        # Limpiar contenedor de etiquetas
        for widget in self.files_container.winfo_children():
            widget.destroy()
            
        for i, (path, info) in enumerate(self.loaded_files.items()):
            full_name = info["name"]
            
            # Al ser tarjetas anchas, ahora podemos permitir que se vea mucho más texto (ej. 80 caracteres)
            display_name = full_name if len(full_name) <= 85 else full_name[:82] + "..."
            
            bg_color = COLORS[i % len(COLORS)]
            fg_color = "black" if bg_color in ["#ff7f0e"] else "white"
            
            # Tarjeta ancha que se estira horizontalmente (fill=tk.X)
            tag = tk.Frame(self.files_container, bg=bg_color, relief=tk.RAISED, bd=1, padx=10, pady=4)
            tag.pack(fill=tk.X, pady=2, padx=2, expand=True)
            
            # El texto se alinea a la izquierda (anchor="w")
            lbl = tk.Label(tag, text=display_name, bg=bg_color, fg=fg_color, font=("Arial", 9, "bold"), anchor="w")
            lbl.pack(side=tk.LEFT, fill=tk.X, expand=True)
            
            ToolTip(lbl, full_name)
            
            btn_del = tk.Button(tag, text="❌", command=lambda p=path: self.remove_file(p), 
                                bg=bg_color, fg=fg_color, bd=0, activebackground=bg_color, 
                                activeforeground="red", cursor="hand2")
            btn_del.pack(side=tk.RIGHT, padx=(5, 0))

    def change_metric(self, metric_id):
        self.current_metric = metric_id
        self.update_plot()

    def update_plot(self):
        self.ax.clear()
        config = METRIC_CONFIG[self.current_metric]
        metric_label = config["label"]
        graph_type = config["type"]
        
        if not self.loaded_files:
            self.ax.text(0.5, 0.5, "Añade archivos CSV para comenzar a visualizar", 
                         ha='center', va='center', fontsize=12, color='gray')
            self.ax.set_axis_off()
            self.canvas.draw()
            return
            
        self.ax.set_axis_on()
        
        if graph_type == "line":
            for i, (path, info) in enumerate(self.loaded_files.items()):
                df = info["df"]
                color = COLORS[i % len(COLORS)]
                
                if 'epoch' in df.columns and self.current_metric in df.columns:
                    self.ax.plot(df['epoch'], df[self.current_metric], marker='o', 
                                 linewidth=2, color=color)
            
            self.ax.set_xlabel("Epoch")
            self.ax.set_ylabel(metric_label)
            self.ax.grid(True, linestyle='--', alpha=0.6)
            
        elif graph_type == "bar":
            names = []
            values = []
            colors = []
            
            for i, (path, info) in enumerate(self.loaded_files.items()):
                df = info["df"]
                if self.current_metric in df.columns:
                    short_name = info["name"] if len(info["name"]) <= 25 else info["name"][:22] + "..."
                    names.append(short_name)
                    values.append(df[self.current_metric].iloc[-1])
                    colors.append(COLORS[i % len(COLORS)])
            
            if values:
                x_positions = range(len(names))
                bars = self.ax.bar(x_positions, values, color=colors, edgecolor='black', alpha=0.8, width=0.4)
                
                self.ax.set_xticks(x_positions)
                self.ax.set_xticklabels(names, rotation=15, ha="right", fontsize=9)
                self.ax.set_ylabel(metric_label)
                self.ax.grid(axis='y', linestyle='--', alpha=0.5)
                self.ax.bar_label(bars, fmt='%.2f', padding=3, fontsize=9)
                
        self.ax.set_title(f"Comparativa: {metric_label}", fontsize=12, fontweight='bold', pad=15)
        self.fig.tight_layout()
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = AppVisualizador(root)
    root.mainloop()