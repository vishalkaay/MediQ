import json, pickle
from pathlib import Path
import pandas as pd
from flask import Flask, request, render_template_string, jsonify

BASE = Path(__file__).resolve().parent
ART = BASE / "artifacts"
app = Flask(__name__)

def obj(d): return type("Obj", (object,), d)

def load():
    model_path, symptoms_path, kb_path = ART/"disease_model.pkl", ART/"symptoms.json", ART/"knowledge_base.json"
    if not model_path.exists(): raise FileNotFoundError("Run python train_model.py first")
    return pickle.load(open(model_path, "rb")), json.load(open(symptoms_path, encoding="utf-8")), json.load(open(kb_path, encoding="utf-8"))

BUNDLE, SYMPTOMS, KB = load()

HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MediQ | Intelligent Symptom Diagnostics</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

    :root {
      --bg-color: #0b0f19;
      --surface-color: #111827;
      --surface-card: #1f2937;
      --accent-primary: #6366f1;
      --accent-secondary: #a855f7;
      --accent-gradient: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
      --accent-gradient-hover: linear-gradient(135deg, #4f46e5 0%, #9333ea 100%);
      --text-main: #f3f4f6;
      --text-muted: #9ca3af;
      --border-color: #374151;
      --severity-low: #3b82f6;
      --severity-med: #f59e0b;
      --severity-high: #ef4444;
      --warning: #f59e0b;
    }

    * {
      box-sizing: border-box;
    }

    body {
      font-family: 'Plus Jakarta Sans', system-ui, -apple-system, sans-serif;
      background-color: var(--bg-color);
      color: var(--text-main);
      margin: 0;
      padding: 0;
      min-height: 100vh;
      background-image: 
        radial-gradient(circle at 10% 20%, rgba(99, 102, 241, 0.15) 0%, transparent 40%),
        radial-gradient(circle at 90% 80%, rgba(168, 85, 247, 0.1) 0%, transparent 40%);
      background-attachment: fixed;
    }

    .app-container {
      max-width: 1280px;
      margin: 0 auto;
      padding: 40px 20px;
    }

    /* Header */
    .app-header {
      margin-bottom: 36px;
      border-bottom: 1px solid var(--border-color);
      padding-bottom: 24px;
    }
    .logo-area {
      display: flex;
      align-items: center;
      gap: 16px;
    }
    .logo-area h1 {
      font-size: 2.2rem;
      font-weight: 800;
      margin: 0;
      letter-spacing: -0.5px;
      background: linear-gradient(to right, #ffffff, #a855f7);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }
    .badge-tech {
      font-size: 0.8rem;
      font-weight: 600;
      text-transform: uppercase;
      background: rgba(99, 102, 241, 0.2);
      color: #818cf8;
      padding: 4px 10px;
      border-radius: 99px;
      letter-spacing: 1px;
      border: 1px solid rgba(99, 102, 241, 0.3);
    }
    .pulse-svg {
      width: 60px;
      height: 28px;
    }
    .pulse-svg path {
      stroke-dasharray: 100;
      stroke-dashoffset: 100;
      animation: drawPulse 2.5s ease-in-out infinite;
    }
    @keyframes drawPulse {
      0% { stroke-dashoffset: 100; }
      50% { stroke-dashoffset: 0; }
      100% { stroke-dashoffset: -100; }
    }
    .subtitle {
      color: var(--text-muted);
      font-size: 1rem;
      margin: 8px 0 0 0;
    }

    /* Dashboard Grid */
    .dashboard {
      display: grid;
      grid-template-columns: 400px 1fr;
      gap: 32px;
      align-items: start;
    }
    @media (max-width: 1024px) {
      .dashboard {
        grid-template-columns: 1fr;
      }
    }

    /* Console Panel (Left) */
    .console-panel {
      background-color: var(--surface-color);
      border: 1px solid var(--border-color);
      border-radius: 16px;
      padding: 24px;
      box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
    }
    .console-panel h2 {
      font-size: 1.3rem;
      font-weight: 700;
      margin-top: 0;
      margin-bottom: 20px;
      color: white;
    }

    .search-container {
      position: relative;
      margin-bottom: 20px;
    }
    .search-box-wrapper {
      position: relative;
      display: flex;
      align-items: center;
    }
    .search-icon {
      position: absolute;
      left: 16px;
      width: 20px;
      height: 20px;
      color: var(--text-muted);
      pointer-events: none;
    }
    #symptom-input {
      width: 100%;
      padding: 16px 40px 16px 48px;
      background-color: rgba(255, 255, 255, 0.02);
      border: 2px solid var(--border-color);
      border-radius: 12px;
      color: var(--text-main);
      font-size: 1rem;
      font-family: inherit;
      transition: all 0.3s ease;
      box-sizing: border-box;
    }
    #symptom-input:focus {
      outline: none;
      border-color: var(--accent-primary);
      box-shadow: 0 0 0 4px rgba(99, 102, 241, 0.15);
      background-color: rgba(255, 255, 255, 0.04);
    }
    .clear-search-btn {
      position: absolute;
      right: 16px;
      background: none;
      border: none;
      color: var(--text-muted);
      font-size: 1.1rem;
      cursor: pointer;
    }
    .clear-search-btn:hover {
      color: var(--text-main);
    }
    .autocomplete-dropdown {
      position: absolute;
      top: 100%;
      left: 0;
      right: 0;
      z-index: 100;
      background-color: var(--surface-color);
      border: 1px solid var(--border-color);
      border-radius: 12px;
      margin-top: 8px;
      max-height: 260px;
      overflow-y: auto;
      box-shadow: 0 10px 25px rgba(0, 0, 0, 0.4);
    }
    .autocomplete-item {
      padding: 12px 16px;
      cursor: pointer;
      display: flex;
      justify-content: space-between;
      align-items: center;
      border-bottom: 1px solid rgba(255, 255, 255, 0.03);
      transition: background-color 0.2s;
      color: var(--text-main);
    }
    .autocomplete-item:hover, .autocomplete-item.active {
      background-color: rgba(99, 102, 241, 0.15);
    }
    .symptom-severity-dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
    }

    .selected-symptoms-section h3 {
      font-size: 0.9rem;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      color: var(--text-muted);
      margin-bottom: 12px;
    }
    .chips-container {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      min-height: 100px;
      padding: 16px;
      background-color: rgba(255, 255, 255, 0.02);
      border: 1px dashed var(--border-color);
      border-radius: 12px;
      align-content: flex-start;
      margin-bottom: 24px;
    }
    .no-symptoms-placeholder {
      color: var(--text-muted);
      font-size: 0.9rem;
      align-self: center;
      text-align: center;
      width: 100%;
      line-height: 1.5;
      padding: 10px;
    }
    .chip {
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 8px 14px;
      background-color: rgba(99, 102, 241, 0.1);
      border: 1px solid rgba(99, 102, 241, 0.25);
      border-radius: 8px;
      font-size: 0.9rem;
      color: var(--text-main);
      animation: popIn 0.2s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }
    @keyframes popIn {
      from { transform: scale(0.8); opacity: 0; }
      to { transform: scale(1); opacity: 1; }
    }
    .chip.sev-1 { border-color: rgba(59, 130, 246, 0.5); background-color: rgba(59, 130, 246, 0.1); }
    .chip.sev-2 { border-color: rgba(59, 130, 246, 0.7); background-color: rgba(59, 130, 246, 0.15); }
    .chip.sev-3 { border-color: rgba(245, 158, 11, 0.5); background-color: rgba(245, 158, 11, 0.1); }
    .chip.sev-4 { border-color: rgba(245, 158, 11, 0.7); background-color: rgba(245, 158, 11, 0.15); }
    .chip.sev-5 { border-color: rgba(239, 68, 68, 0.5); background-color: rgba(239, 68, 68, 0.1); }
    .chip.sev-6 { border-color: rgba(239, 68, 68, 0.7); background-color: rgba(239, 68, 68, 0.15); }
    .chip.sev-7 { border-color: rgba(239, 68, 68, 1); background-color: rgba(239, 68, 68, 0.2); }

    .chip-close {
      background: none;
      border: none;
      color: var(--text-muted);
      cursor: pointer;
      padding: 0;
      display: flex;
      align-items: center;
      font-size: 0.95rem;
    }
    .chip-close:hover {
      color: var(--text-main);
    }

    .action-buttons-group {
      display: flex;
      flex-direction: column;
      gap: 12px;
    }
    .btn-primary {
      width: 100%;
      padding: 16px;
      background: var(--accent-gradient);
      border: none;
      border-radius: 12px;
      color: white;
      font-size: 1rem;
      font-weight: 700;
      cursor: pointer;
      display: flex;
      justify-content: center;
      align-items: center;
      gap: 10px;
      transition: all 0.3s;
      box-shadow: 0 4px 15px rgba(99, 102, 241, 0.35);
    }
    .btn-primary:hover:not(:disabled) {
      background: var(--accent-gradient-hover);
      box-shadow: 0 6px 20px rgba(99, 102, 241, 0.5);
      transform: translateY(-2px);
    }
    .btn-primary:active:not(:disabled) {
      transform: translateY(0);
    }
    .btn-primary:disabled {
      background: var(--border-color);
      color: var(--text-muted);
      cursor: not-allowed;
      box-shadow: none;
    }
    .btn-arrow {
      width: 18px;
      height: 18px;
      transition: transform 0.2s;
      stroke: currentColor;
    }
    .btn-primary:hover .btn-arrow:not(:disabled) {
      transform: translateX(4px);
    }
    .btn-secondary {
      width: 100%;
      padding: 14px;
      background-color: transparent;
      border: 1px solid var(--border-color);
      border-radius: 12px;
      color: var(--text-muted);
      font-size: 0.95rem;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s;
    }
    .btn-secondary:hover {
      border-color: var(--text-muted);
      color: var(--text-main);
      background-color: rgba(255, 255, 255, 0.02);
    }

    /* Results Section (Right) */
    .results-container {
      min-height: 400px;
    }
    .results-card {
      background-color: var(--surface-color);
      border: 1px solid var(--border-color);
      border-radius: 16px;
      padding: 40px;
      text-align: center;
      box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      min-height: 400px;
    }
    .placeholder-icon-wrapper {
      background: rgba(99, 102, 241, 0.1);
      padding: 20px;
      border-radius: 50%;
      margin-bottom: 24px;
    }
    .placeholder-icon {
      width: 48px;
      height: 48px;
      color: var(--accent-primary);
    }
    .placeholder-card h3 {
      font-size: 1.5rem;
      margin: 0 0 12px 0;
      color: white;
    }
    .placeholder-card p {
      color: var(--text-muted);
      max-width: 480px;
      line-height: 1.6;
      margin: 0;
      font-size: 0.95rem;
    }

    .results-layout {
      display: flex;
      flex-direction: column;
      gap: 24px;
    }

    /* Radar loading loader */
    .radar-container {
      position: relative;
      width: 80px;
      height: 80px;
      margin-bottom: 24px;
    }
    .radar-dot {
      position: absolute;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%);
      width: 12px;
      height: 12px;
      background-color: var(--accent-primary);
      border-radius: 50%;
      box-shadow: 0 0 15px var(--accent-primary);
    }
    .radar-ring {
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      border: 2px solid var(--accent-primary);
      border-radius: 50%;
      opacity: 0;
      animation: radarPulse 2s cubic-bezier(0.1, 0.8, 0.3, 1) infinite;
    }
    .radar-ring:nth-child(2) {
      animation-delay: 0.6s;
    }
    .radar-ring:nth-child(3) {
      animation-delay: 1.2s;
    }
    @keyframes radarPulse {
      0% { transform: scale(0.1); opacity: 0; }
      10% { opacity: 0.5; }
      100% { transform: scale(1.2); opacity: 0; }
    }
    .loading-card h3 {
      font-size: 1.4rem;
      color: white;
      margin: 0 0 8px 0;
    }
    .loading-step {
      color: var(--text-muted);
      font-size: 0.95rem;
      margin: 0;
    }

    /* Diagnosis Hero Card */
    .diagnosis-hero {
      background: radial-gradient(circle at 0% 0%, rgba(99, 102, 241, 0.15) 0%, transparent 60%), var(--surface-color);
      border: 1px solid var(--border-color);
      border-radius: 16px;
      padding: 32px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 32px;
      box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
    }
    @media (max-width: 640px) {
      .diagnosis-hero {
        flex-direction: column;
        align-items: flex-start;
      }
      .gauge-container {
        align-self: center;
      }
    }
    .hero-details {
      flex: 1;
    }
    .section-tag {
      font-size: 0.8rem;
      font-weight: 700;
      color: #a855f7;
      text-transform: uppercase;
      letter-spacing: 1px;
    }
    .hero-details h2 {
      font-size: 2.2rem;
      font-weight: 800;
      margin: 8px 0 12px 0;
      color: white;
      letter-spacing: -0.5px;
    }
    .hero-details p {
      color: var(--text-muted);
      line-height: 1.6;
      margin: 0;
      font-size: 1rem;
    }

    .gauge-container {
      position: relative;
      width: 140px;
      height: 140px;
      flex-shrink: 0;
    }
    .gauge {
      transform: rotate(-90deg);
      width: 100%;
      height: 100%;
    }
    .gauge-fill {
      transition: stroke-dashoffset 0.8s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .gauge-text {
      position: absolute;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%);
      text-align: center;
    }
    .gauge-value {
      font-size: 2.2rem;
      font-weight: 800;
      color: var(--text-main);
    }
    .gauge-percent {
      font-size: 1.1rem;
      color: var(--accent-secondary);
      font-weight: 600;
    }

    /* Safety Warning */
    .safety-warning {
      background-color: rgba(245, 158, 11, 0.05);
      border: 1px solid rgba(245, 158, 11, 0.2);
      border-radius: 12px;
      padding: 20px;
      display: flex;
      gap: 16px;
      align-items: flex-start;
      text-align: left;
    }
    .warning-icon {
      width: 24px;
      height: 24px;
      color: var(--warning);
      flex-shrink: 0;
      margin-top: 2px;
    }
    .warning-text-container h4 {
      margin: 0 0 6px 0;
      color: var(--warning);
      font-size: 0.95rem;
      font-weight: 700;
    }
    .warning-text-container p {
      margin: 0;
      color: rgba(245, 158, 11, 0.85);
      font-size: 0.88rem;
      line-height: 1.5;
    }

    /* Info Grid */
    .info-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 20px;
    }
    .info-card {
      background-color: var(--surface-color);
      border: 1px solid var(--border-color);
      border-radius: 16px;
      padding: 24px;
      display: flex;
      flex-direction: column;
      box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
      transition: transform 0.2s, border-color 0.2s;
      text-align: left;
    }
    .info-card:hover {
      border-color: rgba(255, 255, 255, 0.08);
    }
    .card-header {
      display: flex;
      align-items: center;
      gap: 12px;
      margin-bottom: 20px;
    }
    .card-header h3 {
      font-size: 1.1rem;
      font-weight: 700;
      margin: 0;
      color: white;
    }
    .icon-box {
      width: 36px;
      height: 36px;
      border-radius: 8px;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    .icon-box svg {
      width: 20px;
      height: 20px;
      stroke: currentColor;
    }
    .color-precaution { background-color: rgba(16, 185, 129, 0.1); color: var(--severity-low); }
    .color-diet { background-color: rgba(14, 165, 233, 0.1); color: #38bdf8; }
    .color-medication { background-color: rgba(239, 68, 68, 0.1); color: var(--severity-high); }
    .color-lifestyle { background-color: rgba(245, 158, 11, 0.1); color: var(--severity-med); }
    .color-symptoms { background-color: rgba(168, 85, 247, 0.1); color: var(--accent-secondary); }
    .color-severity { background-color: rgba(99, 102, 241, 0.1); color: var(--accent-primary); }

    .bullet-list {
      margin: 0;
      padding-left: 0;
      list-style: none;
      display: flex;
      flex-direction: column;
      gap: 12px;
      flex-grow: 1;
    }
    .bullet-list li {
      position: relative;
      padding-left: 20px;
      font-size: 0.95rem;
      line-height: 1.5;
      color: rgba(255, 255, 255, 0.85);
    }
    .bullet-list li::before {
      content: "•";
      position: absolute;
      left: 4px;
      color: var(--accent-primary);
      font-size: 1.2rem;
      line-height: 1;
    }
    .card-footer-note {
      margin-top: 16px;
      font-size: 0.78rem;
      color: var(--text-muted);
      border-top: 1px solid rgba(255, 255, 255, 0.03);
      padding-top: 10px;
    }

    /* Symptom Tags */
    .symptom-tag-container {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }
    .symptom-tag {
      background-color: rgba(255, 255, 255, 0.03);
      border: 1px solid var(--border-color);
      color: var(--text-muted);
      padding: 6px 12px;
      border-radius: 6px;
      font-size: 0.85rem;
      text-transform: capitalize;
    }

    /* Severity stat card */
    .severity-stat {
      display: flex;
      align-items: baseline;
      gap: 12px;
      margin-bottom: 8px;
    }
    .severity-num {
      font-size: 3rem;
      font-weight: 800;
      color: white;
      line-height: 1;
    }
    .severity-label {
      font-size: 0.9rem;
      color: var(--text-muted);
      font-weight: 500;
    }
    .severity-desc {
      font-size: 0.88rem;
      color: var(--text-muted);
      line-height: 1.5;
      margin: 0;
    }

    /* Top Predictions differential */
    .top-predictions-section {
      background-color: var(--surface-color);
      border: 1px solid var(--border-color);
      border-radius: 16px;
      padding: 24px;
      box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
      text-align: left;
    }
    .top-predictions-section h3 {
      font-size: 1.1rem;
      font-weight: 700;
      margin-top: 0;
      margin-bottom: 20px;
      color: white;
    }
    .predictions-bars-list {
      display: flex;
      flex-direction: column;
      gap: 16px;
    }
    .prediction-row {
      display: flex;
      align-items: center;
      gap: 16px;
    }
    .pred-label {
      width: 240px;
      font-size: 0.95rem;
      font-weight: 600;
      color: rgba(255,255,255,0.9);
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .pred-bar-wrapper {
      flex: 1;
      height: 12px;
      background-color: rgba(255, 255, 255, 0.05);
      border-radius: 6px;
      overflow: hidden;
      position: relative;
    }
    .pred-bar-fill {
      height: 100%;
      background: var(--accent-gradient);
      border-radius: 6px;
      transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
      width: 0%;
    }
    .pred-value {
      width: 60px;
      text-align: right;
      font-size: 0.95rem;
      font-weight: 700;
      color: var(--accent-primary);
    }
  </style>
</head>
<body>
  <div class="app-container">
    <header class="app-header">
      <div class="logo-area">
        <div class="logo-icon-pulse">
          <svg class="pulse-svg" viewBox="0 0 100 40">
            <path d="M 0,20 L 30,20 L 40,5 L 50,35 L 60,15 L 70,25 L 75,20 L 100,20" fill="none" stroke="#6366f1" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>
          </svg>
        </div>
        <h1>MediQ <span class="badge-tech">Diagnostics</span></h1>
      </div>
      <p class="subtitle">Clinical-grade predictive intelligence system for symptom-based health analysis</p>
    </header>

    <main class="dashboard">
      <!-- Left Column Console -->
      <section class="console-panel">
        <h2>Diagnostic Console</h2>
        
        <div class="search-container">
          <div class="search-box-wrapper">
            <svg class="search-icon" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2.5" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
            </svg>
            <input type="text" id="symptom-input" placeholder="Search symptoms (e.g. fever)..." autocomplete="off">
            <button id="btn-clear-search" class="clear-search-btn" style="display:none;">✕</button>
          </div>
          
          <div id="autocomplete-list" class="autocomplete-dropdown" style="display:none;"></div>
        </div>

        <div class="selected-symptoms-section">
          <h3>Selected Symptoms (<span id="symptoms-count">0</span>)</h3>
          <div id="selected-chips" class="chips-container">
            <div class="no-symptoms-placeholder">No symptoms selected. Search and add symptoms to begin diagnostic modeling.</div>
          </div>
        </div>

        <div class="action-buttons-group">
          <button id="btn-submit" class="btn-primary" disabled>
            <span>Run Diagnostic Model</span>
            <svg class="btn-arrow" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2.5" d="M14 5l7 7m0 0l-7 7m7-7H3"/>
            </svg>
          </button>
          <button id="btn-clear-all" class="btn-secondary" style="display:none;">Clear All Selected</button>
        </div>
      </section>

      <!-- Right Column Results -->
      <section class="results-container">
        
        <!-- Empty State Placehoder -->
        <div id="results-placeholder" class="results-card placeholder-card">
          <div class="placeholder-icon-wrapper">
            <svg class="placeholder-icon" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
          </div>
          <h3>Awaiting Input</h3>
          <p>Select one or more symptoms from the diagnostic console and click "Run Diagnostic Model" to compute probability weights and treatment protocols.</p>
        </div>

        <!-- Loading State -->
        <div id="results-loading" class="results-card loading-card" style="display:none;">
          <div class="radar-container">
            <div class="radar-ring"></div>
            <div class="radar-ring"></div>
            <div class="radar-ring"></div>
            <div class="radar-dot"></div>
          </div>
          <h3>Analyzing Symptoms</h3>
          <p class="loading-step">Running Random Forest Classifier...</p>
        </div>

        <!-- Results Display Layout -->
        <div id="results-display" class="results-layout" style="display:none;">
          
          <!-- Diagnosis Hero -->
          <div class="diagnosis-hero">
            <div class="hero-details">
              <span class="section-tag">Primary Diagnosis</span>
              <h2 id="diag-disease">Disease Name</h2>
              <p id="diag-description">Description of the disease goes here...</p>
            </div>
            
            <div class="gauge-container">
              <svg class="gauge" viewBox="0 0 100 100">
                <defs>
                  <linearGradient id="gauge-gradient" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" stop-color="#6366f1" />
                    <stop offset="100%" stop-color="#a855f7" />
                  </linearGradient>
                </defs>
                <circle class="gauge-bg" cx="50" cy="50" r="42" fill="none" stroke="rgba(255,255,255,0.05)" stroke-width="8"></circle>
                <circle id="diag-gauge-fill" class="gauge-fill" cx="50" cy="50" r="42" fill="none" stroke="url(#gauge-gradient)" stroke-width="8" stroke-dasharray="264" stroke-dashoffset="264" stroke-linecap="round"></circle>
              </svg>
              <div class="gauge-text">
                <span id="diag-confidence" class="gauge-value">0</span><span class="gauge-percent">%</span>
              </div>
            </div>
          </div>

          <!-- Disclaimer Warning -->
          <div class="safety-warning">
            <svg class="warning-icon" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
            </svg>
            <div class="warning-text-container">
              <h4>Medical Safety Disclaimer</h4>
              <p>This diagnosis is generated by an algorithmic model trained on historical data. It is for educational purposes only. For symptoms like severe breathing difficulties, chest pain, high fever, or stroke-like signs, consult emergency services immediately.</p>
            </div>
          </div>

          <!-- Cards Grid -->
          <div class="info-grid">
            
            <!-- Precautions -->
            <div class="info-card">
              <div class="card-header">
                <div class="icon-box color-precaution">
                  <svg fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z"/></svg>
                </div>
                <h3>Recommended Precautions</h3>
              </div>
              <ul id="card-precautions" class="bullet-list"></ul>
            </div>

            <!-- Diets -->
            <div class="info-card">
              <div class="card-header">
                <div class="icon-box color-diet">
                  <svg fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 6l3 1m0 0l-3 9a5.002 5.002 0 006.001 0M6 7l3 9M6 7l6-2m6 2l3-1m-3 1l-3 9a5.002 5.002 0 006.001 0M18 7l3 9m-3-9l-6-2m0-2v2m0 16V5m0 16H9m3 0h3"/></svg>
                </div>
                <h3>Dietary Guidelines</h3>
              </div>
              <ul id="card-diets" class="bullet-list"></ul>
            </div>

            <!-- Medications -->
            <div class="info-card">
              <div class="card-header">
                <div class="icon-box color-medication">
                  <svg fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z"/></svg>
                </div>
                <h3>Medication Overview</h3>
              </div>
              <ul id="card-medications" class="bullet-list"></ul>
              <div class="card-footer-note">Information purposes only — consult a doctor.</div>
            </div>

            <!-- Lifestyle -->
            <div class="info-card">
              <div class="card-header">
                <div class="icon-box color-lifestyle">
                  <svg fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"/></svg>
                </div>
                <h3>Lifestyle & Workout</h3>
              </div>
              <ul id="card-lifestyle" class="bullet-list"></ul>
            </div>

            <!-- Common Symptoms -->
            <div class="info-card">
              <div class="card-header">
                <div class="icon-box color-symptoms">
                  <svg fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"/></svg>
                </div>
                <h3>Common Symptoms</h3>
              </div>
              <div id="card-common-symptoms" class="symptom-tag-container"></div>
            </div>

            <!-- Severity burden -->
            <div class="info-card severity-card">
              <div class="card-header">
                <div class="icon-box color-severity">
                  <svg fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"/></svg>
                </div>
                <h3>Symptom Burden</h3>
              </div>
              <div class="severity-stat">
                <div id="diag-severity-score" class="severity-num">0</div>
                <div class="severity-label">Cumulative Weight</div>
              </div>
              <p class="severity-desc">Aggregated rating of selected symptoms. Higher scores indicate potentially higher acute discomfort.</p>
            </div>

          </div>

          <!-- Top Predictions comparison -->
          <div class="top-predictions-section">
            <h3>Differential Diagnoses (Top 5 Probabilities)</h3>
            <div id="top-predictions-list" class="predictions-bars-list"></div>
          </div>

        </div>

      </section>
    </main>
  </div>

  <script>
    document.addEventListener('DOMContentLoaded', () => {
      const allSymptoms = {{ symptoms_json | safe }};
      let selectedSymptoms = [];

      const symptomInput = document.getElementById('symptom-input');
      const autocompleteList = document.getElementById('autocomplete-list');
      const selectedChips = document.getElementById('selected-chips');
      const symptomsCount = document.getElementById('symptoms-count');
      const btnSubmit = document.getElementById('btn-submit');
      const btnClearSearch = document.getElementById('btn-clear-search');
      const btnClearAll = document.getElementById('btn-clear-all');

      const placeholderCard = document.getElementById('results-placeholder');
      const loadingCard = document.getElementById('results-loading');
      const resultsDisplay = document.getElementById('results-display');

      let activeIndex = -1;

      // Render selected chips
      function renderChips() {
        selectedChips.innerHTML = '';
        if (selectedSymptoms.length === 0) {
          selectedChips.innerHTML = '<div class="no-symptoms-placeholder">No symptoms selected. Search and add symptoms to begin diagnostic modeling.</div>';
          symptomsCount.textContent = '0';
          btnSubmit.disabled = true;
          btnClearAll.style.display = 'none';
          return;
        }

        symptomsCount.textContent = selectedSymptoms.length;
        btnSubmit.disabled = false;
        btnClearAll.style.display = 'block';

        selectedSymptoms.forEach(sym => {
          const chip = document.createElement('div');
          chip.className = `chip sev-${sym.severity || 1}`;
          
          const label = document.createElement('span');
          label.textContent = sym.label;
          
          const closeBtn = document.createElement('button');
          closeBtn.className = 'chip-close';
          closeBtn.innerHTML = '✕';
          closeBtn.addEventListener('click', () => {
            removeSymptom(sym.id);
          });
          
          chip.appendChild(label);
          chip.appendChild(closeBtn);
          selectedChips.appendChild(chip);
        });
      }

      function addSymptom(sym) {
        if (!selectedSymptoms.some(s => s.id === sym.id)) {
          selectedSymptoms.push(sym);
          renderChips();
        }
        symptomInput.value = '';
        btnClearSearch.style.display = 'none';
        closeDropdown();
      }

      function removeSymptom(id) {
        selectedSymptoms = selectedSymptoms.filter(s => s.id !== id);
        renderChips();
      }

      function closeDropdown() {
        autocompleteList.style.display = 'none';
        autocompleteList.innerHTML = '';
        activeIndex = -1;
      }

      function filterSymptoms(query) {
        if (!query) {
          closeDropdown();
          return;
        }
        
        const filtered = allSymptoms.filter(sym => {
          const isAlreadySelected = selectedSymptoms.some(s => s.id === sym.id);
          const matchesQuery = sym.label.toLowerCase().includes(query.toLowerCase());
          return !isAlreadySelected && matchesQuery;
        });

        if (filtered.length === 0) {
          autocompleteList.innerHTML = '<div style="padding:12px 16px; color:var(--text-muted); font-size:0.9rem;">No matching symptoms found</div>';
          autocompleteList.style.display = 'block';
          return;
        }

        autocompleteList.innerHTML = '';
        filtered.forEach((sym, index) => {
          const item = document.createElement('div');
          item.className = 'autocomplete-item';
          if (index === activeIndex) item.classList.add('active');
          
          const textSpan = document.createElement('span');
          textSpan.textContent = sym.label;
          
          const dot = document.createElement('div');
          dot.className = 'symptom-severity-dot';
          let dotColor = 'var(--severity-low)';
          if (sym.severity >= 5) dotColor = 'var(--severity-high)';
          else if (sym.severity >= 3) dotColor = 'var(--severity-med)';
          dot.style.backgroundColor = dotColor;
          
          item.appendChild(textSpan);
          item.appendChild(dot);
          
          item.addEventListener('click', () => {
            addSymptom(sym);
          });
          
          autocompleteList.appendChild(item);
        });
        autocompleteList.style.display = 'block';
      }

      symptomInput.addEventListener('input', (e) => {
        const val = e.target.value;
        btnClearSearch.style.display = val ? 'block' : 'none';
        filterSymptoms(val);
      });

      symptomInput.addEventListener('focus', () => {
        filterSymptoms(symptomInput.value);
      });

      document.addEventListener('click', (e) => {
        if (!symptomInput.contains(e.target) && !autocompleteList.contains(e.target)) {
          closeDropdown();
        }
      });

      btnClearSearch.addEventListener('click', () => {
        symptomInput.value = '';
        btnClearSearch.style.display = 'none';
        closeDropdown();
        symptomInput.focus();
      });

      btnClearAll.addEventListener('click', () => {
        selectedSymptoms = [];
        renderChips();
      });

      symptomInput.addEventListener('keydown', (e) => {
        const items = autocompleteList.querySelectorAll('.autocomplete-item');
        if (items.length === 0) return;

        if (e.key === 'ArrowDown') {
          e.preventDefault();
          activeIndex = (activeIndex + 1) % items.length;
          updateActiveItem(items);
        } else if (e.key === 'ArrowUp') {
          e.preventDefault();
          activeIndex = (activeIndex - 1 + items.length) % items.length;
          updateActiveItem(items);
        } else if (e.key === 'Enter') {
          e.preventDefault();
          if (activeIndex >= 0 && activeIndex < items.length) {
            items[activeIndex].click();
          } else if (items.length > 0) {
            items[0].click();
          }
        } else if (e.key === 'Escape') {
          closeDropdown();
        }
      });

      function updateActiveItem(items) {
        items.forEach((item, idx) => {
          if (idx === activeIndex) {
            item.classList.add('active');
            item.scrollIntoView({ block: 'nearest' });
          } else {
            item.classList.remove('active');
          }
        });
      }

      btnSubmit.addEventListener('click', async () => {
        if (selectedSymptoms.length === 0) return;

        placeholderCard.style.display = 'none';
        resultsDisplay.style.display = 'none';
        loadingCard.style.display = 'flex';

        const symptomsIds = selectedSymptoms.map(s => s.id);

        try {
          const response = await fetch('/predict', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
              'Accept': 'application/json'
            },
            body: JSON.stringify({ symptoms: symptomsIds })
          });

          if (!response.ok) throw new Error('Prediction API failed');

          const data = await response.json();
          displayResults(data);
        } catch (err) {
          console.error(err);
          alert('Failed to connect to the diagnostic server. Please try again.');
          loadingCard.style.display = 'none';
          placeholderCard.style.display = 'flex';
        }
      });

      function displayResults(data) {
        const { result, top } = data;

        document.getElementById('diag-disease').textContent = result.disease;
        document.getElementById('diag-description').textContent = result.description || 'No description available for this disease.';
        document.getElementById('diag-confidence').textContent = result.confidence;

        const fill = document.getElementById('diag-gauge-fill');
        const confidence = parseFloat(result.confidence);
        const offset = 264 - (264 * confidence / 100);
        fill.style.strokeDashoffset = offset;

        populateList('card-precautions', result.precautions, 'No specific precautions listed.');
        populateList('card-diets', result.diets, 'No dietary guidelines available.');
        populateList('card-medications', result.medications, 'No medication information listed.');
        populateList('card-lifestyle', result.lifestyle_recommendations, 'No lifestyle recommendations available.');

        const commonContainer = document.getElementById('card-common-symptoms');
        commonContainer.innerHTML = '';
        if (result.common_symptoms && result.common_symptoms.length > 0) {
          result.common_symptoms.slice(0, 12).forEach(sym => {
            const tag = document.createElement('span');
            tag.className = 'symptom-tag';
            tag.textContent = sym.replace(/_/g, ' ');
            commonContainer.appendChild(tag);
          });
        } else {
          commonContainer.innerHTML = '<div style="color:var(--text-muted); font-size:0.9rem;">No common symptoms data.</div>';
        }

        document.getElementById('diag-severity-score').textContent = result.severity_score;

        const predictionsList = document.getElementById('top-predictions-list');
        predictionsList.innerHTML = '';
        
        top.forEach(item => {
          const row = document.createElement('div');
          row.className = 'prediction-row';
          
          const label = document.createElement('div');
          label.className = 'pred-label';
          label.textContent = item.disease;
          
          const barWrapper = document.createElement('div');
          barWrapper.className = 'pred-bar-wrapper';
          
          const barFill = document.createElement('div');
          barFill.className = 'pred-bar-fill';
          
          const val = document.createElement('div');
          val.className = 'pred-value';
          val.textContent = `${item.probability}%`;
          
          barWrapper.appendChild(barFill);
          row.appendChild(label);
          row.appendChild(barWrapper);
          row.appendChild(val);
          
          predictionsList.appendChild(row);

          setTimeout(() => {
            barFill.style.width = `${item.probability}%`;
          }, 50);
        });

        loadingCard.style.display = 'none';
        resultsDisplay.style.display = 'flex';

        if (window.innerWidth <= 1024) {
          resultsDisplay.scrollIntoView({ behavior: 'smooth' });
        }
      }

      function populateList(elementId, items, emptyText) {
        const list = document.getElementById(elementId);
        list.innerHTML = '';
        if (items && items.length > 0) {
          items.forEach(item => {
            const li = document.createElement('li');
            li.textContent = item;
            list.appendChild(li);
          });
        } else {
          list.innerHTML = `<li style="list-style:none; color:var(--text-muted); padding-left:0; font-size:0.9rem;">${emptyText}</li>`;
        }
      }
    });
  </script>
</body>
</html>
"""

def predict_selected(selected):
    features, model, le, sev = BUNDLE["feature_cols"], BUNDLE["model"], BUNDLE["label_encoder"], BUNDLE.get("severity", {})
    row = {f: int(f in set(selected)) for f in features}
    X = pd.DataFrame([row], columns=features)
    probs = model.predict_proba(X)[0]
    names = le.inverse_transform(range(len(probs)))
    ranked = sorted(zip(names, probs), key=lambda x: x[1], reverse=True)
    key, prob = ranked[0]
    info = KB.get(key, {})
    result = {
        "disease": info.get("disease", key.title()), "confidence": round(float(prob)*100, 2),
        "description": info.get("description", ""), "precautions": info.get("precautions", []),
        "diets": info.get("diets", []), "medications": info.get("medications", []),
        "lifestyle_recommendations": info.get("lifestyle_recommendations", []), "common_symptoms": info.get("common_symptoms", []),
        "severity_score": sum(int(sev.get(s, 0)) for s in selected), "selected_labels": [s.replace("_", " ") for s in selected]
    }
    top = [{"disease": KB.get(k, {}).get("disease", k.title()), "probability": round(float(p)*100, 2)} for k,p in ranked[:5]]
    return result, top

@app.route('/')
def home():
    return render_template_string(HTML, symptoms=SYMPTOMS, symptoms_json=json.dumps(SYMPTOMS))

@app.route('/predict', methods=['POST'])
def predict():
    if request.is_json:
        data = request.get_json() or {}
        selected = data.get('symptoms', [])
    else:
        selected = request.form.getlist('symptoms')
    result, top = predict_selected(selected)
    return jsonify({"result": result, "top": top})

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5001)
