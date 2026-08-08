// --- Dark Mode ---
// Καθαρά αισθητική επιλογή: θυμάται την προτίμηση του χρήστη (localStorage),
// αλλιώς ακολουθεί την προτίμηση συστήματος (prefers-color-scheme) την πρώτη φορά.
function applyTheme(theme) {
    document.body.classList.toggle('dark', theme === 'dark');
    const icon = document.getElementById('themeIcon');
    if (icon) icon.className = theme === 'dark' ? 'fa-solid fa-sun' : 'fa-solid fa-moon';
}
function toggleTheme() {
    const next = document.body.classList.contains('dark') ? 'light' : 'dark';
    applyTheme(next);
    localStorage.setItem('fruitAppTheme', next);
}
(function initTheme() {
    const saved = localStorage.getItem('fruitAppTheme');
    const prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
    applyTheme(saved || (prefersDark ? 'dark' : 'light'));
})();

// Το backend σερβίρει πλέον και το frontend (main.py -> StaticFiles), οπότε
// τα requests είναι πάντα same-origin - δεν χρειάζεται πλήρες URL, ούτε τοπικά
// ούτε σε production.
const API_URL = "";
let selectedFiles = [];
let videoStream = null;
let streamInterval = null;
let isStreaming = false;

const fileInput = document.getElementById('fileInput');
const previewContainer = document.getElementById('previewContainer');
const previewImage = document.getElementById('previewImage');
const predictBtn = document.getElementById('predictBtn');
const webcamVideo = document.getElementById('webcamVideo');
const captureCanvas = document.getElementById('captureCanvas');
const dropZone = document.getElementById('dropZone');
const batchList = document.getElementById('batchList');

// --- Toast notifications ---
const toastContainer = document.getElementById('toastContainer');
const TOAST_ICONS = { success: 'fa-circle-check', error: 'fa-circle-exclamation', info: 'fa-circle-info' };
function showToast(message, type = 'info', duration = 3800) {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerHTML = `
        <i class="fa-solid ${TOAST_ICONS[type] || TOAST_ICONS.info}"></i>
        <span>${message}</span>
        <button class="toast-close" aria-label="Κλείσιμο">&times;</button>
    `;
    const remove = () => {
        toast.classList.add('leaving');
        setTimeout(() => toast.remove(), 200);
    };
    toast.querySelector('.toast-close').addEventListener('click', remove);
    toastContainer.appendChild(toast);
    setTimeout(remove, duration);
}

// --- Custom confirm dialog (αντικαθιστά το window.confirm) ---
const confirmOverlay = document.getElementById('confirmOverlay');
const confirmMessage = document.getElementById('confirmMessage');
const confirmOkBtn = document.getElementById('confirmOkBtn');
const confirmCancelBtn = document.getElementById('confirmCancelBtn');
function confirmDialog(message) {
    return new Promise((resolve) => {
        confirmMessage.textContent = message;
        confirmOverlay.classList.add('active');
        const cleanup = (result) => {
            confirmOverlay.classList.remove('active');
            confirmOkBtn.removeEventListener('click', onOk);
            confirmCancelBtn.removeEventListener('click', onCancel);
            confirmOverlay.removeEventListener('click', onOverlayClick);
            resolve(result);
        };
        const onOk = () => cleanup(true);
        const onCancel = () => cleanup(false);
        const onOverlayClick = (e) => { if (e.target === confirmOverlay) cleanup(false); };
        confirmOkBtn.addEventListener('click', onOk);
        confirmCancelBtn.addEventListener('click', onCancel);
        confirmOverlay.addEventListener('click', onOverlayClick);
    });
}

// Απλά, μονόχρωμα botanical εικονίδια ανά κατηγορία φρούτου/λαχανικού
// (βασισμένα στο πεδίο "category" της εγκυκλοπαίδειας) - γραφικό στοιχείο
// βγαλμένο από τα ίδια τα δεδομένα, όχι απλή διακόσμηση.
const CATEGORY_ICONS = [
    ['εσπεριδοειδ', '<circle cx="12" cy="12" r="8"/><path d="M12 4L12 20M18.93 8L5.07 16M5.07 8L18.93 16"/>'],
    ['μηλοειδ', '<path d="M12 8c-2.5-2-6-1-6 3.2C6 15.5 9 20 12 20s6-4.5 6-8.8C18 7 14.5 6 12 8z"/><path d="M12 8V5"/><path d="M12 5c0-1 1-2 2-2"/>'],
    ['πυρηνόκαρπ', '<circle cx="12" cy="12" r="8"/><path d="M10.5 4.3c-1.2 3-1.2 12.4 0 15.4"/>'],
    ['μούρο', '<circle cx="9" cy="10" r="3.2"/><circle cx="15" cy="10" r="3.2"/><circle cx="12" cy="15.5" r="3.4"/><path d="M12 6V4"/>'],
    ['τροπικό', '<path d="M4 20c8-1 14-7 15-16-9 1-15 7-16 16z"/><path d="M6 18c3-4 7-8 11-11"/>'],
    ['πεπονοειδ', '<circle cx="12" cy="12" r="8"/><path d="M12 4c-1 4-1 12 0 16"/><path d="M5 10c4 2 10 2 14 0"/><path d="M5 14c4-2 10-2 14 0"/>'],
    ['φρούτο με σπόρους', '<circle cx="12" cy="13" r="7"/><path d="M9 6l1-2h4l1 2"/>'],
    ['ξηρός καρπός', '<ellipse cx="12" cy="12" rx="5.5" ry="8"/><path d="M12 4c-1 4-1 12 0 16"/>'],
    ['λαχανικ', '<path d="M12 5c2 4 3 8 1.5 12.5-0.5 1.5-2.5 1.5-3 0C9 13 10 9 12 5z"/><path d="M9 5l1.5 2"/><path d="M15 5l-1.5 2"/>'],
    ['σταυρανθ', '<path d="M12 5c2 4 3 8 1.5 12.5-0.5 1.5-2.5 1.5-3 0C9 13 10 9 12 5z"/><path d="M9 5l1.5 2"/><path d="M15 5l-1.5 2"/>'],
    ['δημητριακ', '<path d="M12 5c2 4 3 8 1.5 12.5-0.5 1.5-2.5 1.5-3 0C9 13 10 9 12 5z"/><path d="M9 5l1.5 2"/><path d="M15 5l-1.5 2"/>'],
];
const DEFAULT_ICON = '<path d="M4 20c9-1 15-8 16-16-8 1-15 7-16 16z"/><path d="M7 17c3-4 7-8 12-12"/>';

function getCategoryIconMarkup(categoryText) {
    const normalized = (categoryText || '').toLowerCase();
    for (const [keyword, markup] of CATEGORY_ICONS) {
        if (normalized.includes(keyword)) return markup;
    }
    return DEFAULT_ICON;
}

function updateGauge(pct) {
    const circumference = 251.2; // 2 * PI * r(40)
    const clamped = Math.max(0, Math.min(100, pct));
    const offset = circumference - (clamped / 100) * circumference;
    const gaugeFill = document.getElementById('gaugeFill');
    gaugeFill.style.strokeDashoffset = offset;
    gaugeFill.style.stroke = clamped < 40 ? 'var(--danger)' : (clamped < 70 ? 'var(--citrus)' : 'var(--leaf)');
    document.getElementById('gaugePct').textContent = clamped.toFixed(0) + '%';
}

function switchMode(mode) {
    stopLiveStream();
    if (mode === 'upload') {
        document.getElementById('uploadTabBtn').classList.add('active');
        document.getElementById('cameraTabBtn').classList.remove('active');
        document.getElementById('uploadSection').style.display = 'block';
        document.getElementById('cameraSection').style.display = 'none';
        if (videoStream) videoStream.getTracks().forEach(t => t.stop());
    } else {
        document.getElementById('cameraTabBtn').classList.add('active');
        document.getElementById('uploadTabBtn').classList.remove('active');
        document.getElementById('uploadSection').style.display = 'none';
        document.getElementById('cameraSection').style.display = 'block';
        initWebcam();
    }
}

async function initWebcam() {
    try {
        videoStream = await navigator.mediaDevices.getUserMedia({ video: true });
        webcamVideo.srcObject = videoStream;
    } catch (err) {
        showToast("Σφάλμα πρόσβασης στην κάμερα: " + err.message, 'error');
    }
}

function toggleLiveStream() {
    const btn = document.getElementById('streamToggleBtn');
    if (!isStreaming) {
        isStreaming = true;
        btn.innerHTML = `<i class="fa-solid fa-stop"></i> Παύση Stream`;
        btn.style.backgroundColor = '#dc2626';
        streamInterval = setInterval(sendWebcamFrame, 800);
    } else {
        stopLiveStream();
    }
}

function stopLiveStream() {
    isStreaming = false;
    clearInterval(streamInterval);
    const btn = document.getElementById('streamToggleBtn');
    if (btn) {
        btn.innerHTML = `<i class="fa-solid fa-play"></i> Έναρξη Live Stream`;
        btn.style.backgroundColor = '#2563eb';
    }
}

function sendWebcamFrame() {
    if (!webcamVideo.videoWidth) return;
    captureCanvas.width = webcamVideo.videoWidth;
    captureCanvas.height = webcamVideo.videoHeight;
    const ctx = captureCanvas.getContext('2d');
    ctx.drawImage(webcamVideo, 0, 0);

    captureCanvas.toBlob(blob => {
        processPrediction(blob, { isLiveFrame: true, filename: 'frame.jpg' });
    }, 'image/jpeg', 0.8);
}

function handleFilesSelected(fileList) {
    const files = Array.from(fileList).filter(f => f.type.startsWith('image/'));
    if (files.length === 0) return;
    selectedFiles = files;

    if (files.length === 1) {
        previewImage.src = URL.createObjectURL(files[0]);
        previewContainer.style.display = 'block';
        batchList.style.display = 'none';
        batchList.innerHTML = '';
    } else {
        previewContainer.style.display = 'none';
        renderBatchList();
    }

    updatePredictBtnLabel();
    predictBtn.disabled = false;
}

function updatePredictBtnLabel() {
    const label = document.getElementById('btnText');
    label.innerHTML = selectedFiles.length > 1
        ? `<i class="fa-solid fa-wand-magic-sparkles"></i> Εκτέλεση Αναγνώρισης (${selectedFiles.length} εικόνες)`
        : `<i class="fa-solid fa-wand-magic-sparkles"></i> Εκτέλεση Αναγνώρισης`;
}

function renderBatchList() {
    batchList.style.display = 'flex';
    batchList.innerHTML = selectedFiles.map((f, i) => `
        <div class="batch-item status-pending" id="batchItem-${i}">
            <span class="batch-name">${f.name}</span>
            <span class="batch-status" id="batchStatus-${i}"><i class="fa-regular fa-clock"></i></span>
        </div>
    `).join('');
}

function setBatchItemStatus(index, status) {
    const item = document.getElementById(`batchItem-${index}`);
    const statusEl = document.getElementById(`batchStatus-${index}`);
    if (!item || !statusEl) return;
    item.className = `batch-item status-${status}`;
    if (status === 'processing') statusEl.innerHTML = `<i class="fa-solid fa-circle-notch"></i>`;
    else if (status === 'done') statusEl.innerHTML = `<i class="fa-solid fa-check"></i>`;
    else if (status === 'error') statusEl.innerHTML = `<i class="fa-solid fa-xmark"></i>`;
}

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) handleFilesSelected(e.target.files);
});

// Το κείμενο του dropZone υπόσχεται "σύρετε εικόνα εδώ" - ενεργοποιούμε
// πραγματικά το drag & drop, με οπτικό feedback όσο σέρνεις πάνω του.
['dragover', 'dragenter'].forEach(evt => {
    dropZone.addEventListener(evt, (e) => {
        e.preventDefault();
        dropZone.classList.add('drag-active');
    });
});
['dragleave', 'dragend'].forEach(evt => {
    dropZone.addEventListener(evt, () => dropZone.classList.remove('drag-active'));
});
dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('drag-active');
    if (e.dataTransfer.files.length > 0) handleFilesSelected(e.dataTransfer.files);
});

async function uploadAndPredict() {
    if (selectedFiles.length === 0) return;

    document.getElementById('btnText').style.display = 'none';
    document.getElementById('btnSpinner').style.display = 'block';
    predictBtn.disabled = true;

    if (selectedFiles.length === 1) {
        await processPrediction(selectedFiles[0]);
    } else {
        await runBatchPredict();
    }

    document.getElementById('btnText').style.display = 'block';
    document.getElementById('btnSpinner').style.display = 'none';
    predictBtn.disabled = false;
}

async function runBatchPredict() {
    let successCount = 0;
    let errorCount = 0;

    for (let i = 0; i < selectedFiles.length; i++) {
        setBatchItemStatus(i, 'processing');
        try {
            const data = await predictImage(selectedFiles[i]);
            renderPredictionResult(data);
            setBatchItemStatus(i, 'done');
            successCount++;
        } catch (error) {
            console.error(error);
            setBatchItemStatus(i, 'error');
            errorCount++;
        }
    }

    loadHistory();

    if (errorCount === 0) {
        showToast(`Ολοκληρώθηκαν ${successCount} από ${selectedFiles.length} εικόνες με επιτυχία.`, 'success');
    } else {
        showToast(`Ολοκληρώθηκαν ${successCount}/${selectedFiles.length} εικόνες. ${errorCount} απέτυχαν.`, 'error');
    }
}

// Καλεί το /predict και επιστρέφει τα δεδομένα, χωρίς να αγγίζει το DOM
// (χρησιμοποιείται και από το single-upload flow και από το batch flow).
async function predictImage(fileOrBlob, options = {}) {
    const { isLiveFrame = false, filename } = options;
    const formData = new FormData();
    formData.append('file', fileOrBlob, filename || fileOrBlob.name || 'upload.jpg');

    // Τα live camera frames ΔΕΝ αποθηκεύονται στο ιστορικό (save_to_history=false),
    // αλλιώς κάθε 800ms θα δημιουργούσε νέα εγγραφή στη βάση δεδομένων.
    const url = isLiveFrame
        ? `${API_URL}/predict?save_to_history=false`
        : `${API_URL}/predict`;
    const response = await fetch(url, { method: 'POST', body: formData });
    if (!response.ok) throw new Error('Σφάλμα επεξεργασίας.');
    return response.json();
}

function renderPredictionResult(data) {
    document.getElementById('placeholder').style.display = 'none';
    document.getElementById('resultContent').style.display = 'block';

    // Fade/slide-in κάθε φορά που αλλάζει η κύρια πρόβλεψη
    const fruitTagEl = document.querySelector('.fruit-tag');
    fruitTagEl.classList.remove('fade-update');
    void fruitTagEl.offsetWidth; // reflow ώστε να ξαναπαίξει το animation
    fruitTagEl.classList.add('fade-update');

    document.getElementById('fruitResult').textContent = data.prediction;
    document.getElementById('categoryIcon').innerHTML = getCategoryIconMarkup(data.info.category);
    document.getElementById('categoryLabel').textContent = data.info.category;
    updateGauge(data.confidence_percentage);

    // Ίδιο κατώφλι (40%) με το χρώμα του gauge - αν το gauge δείχνει κόκκινο,
    // δείχνουμε και ρητή προειδοποίηση χαμηλής βεβαιότητας.
    document.getElementById('lowConfidenceWarning').style.display =
        data.confidence_percentage < 40 ? 'flex' : 'none';

    // Grad-CAM: επιστρέφεται μόνο για κανονικά uploads (όχι live camera frames)
    const gradcamBox = document.getElementById('gradcamBox');
    if (data.gradcam_image) {
        document.getElementById('gradcamImage').src = data.gradcam_image;
        gradcamBox.style.display = 'block';
    } else {
        gradcamBox.style.display = 'none';
    }

    const top3List = document.getElementById('top3List');
    top3List.innerHTML = '';
    data.top3.forEach((item, index) => {
        top3List.innerHTML += `
            <div class="top3-item" style="animation-delay: ${index * 0.08}s;">
                <div class="top3-label">
                    <span>${index + 1}. ${item.fruit}</span>
                    <span>${item.confidence}%</span>
                </div>
                <div class="progress-bar-bg">
                    <div class="progress-bar-fill" style="width: ${item.confidence}%;"></div>
                </div>
            </div>
        `;
    });

    document.getElementById('infoCard').style.display = 'block';
    document.getElementById('infoCategory').textContent = data.info.category;
    document.getElementById('infoOrigin').textContent = data.info.origin;
    document.getElementById('infoSeason').textContent = data.info.season;
    document.getElementById('infoUses').textContent = data.info.uses;
    document.getElementById('infoFunFact').textContent = data.info.fun_fact;
}

async function processPrediction(fileOrBlob, options = {}) {
    const { isLiveFrame = false, filename } = options;
    try {
        const data = await predictImage(fileOrBlob, { isLiveFrame, filename });
        renderPredictionResult(data);

        // Το ιστορικό ανανεώνεται μόνο για upload (όχι για κάθε live camera frame)
        if (!isLiveFrame) {
            loadHistory();
        }
    } catch (error) {
        console.error(error);
        if (!isLiveFrame) {
            showToast('Απέτυχε η επεξεργασία της εικόνας.', 'error');
        }
    }
}

// --- Ιστορικό: αποθηκεύουμε όλα τα records τοπικά και κάνουμε
// αναζήτηση/φίλτρο/ταξινόμηση/pagination client-side, αφού το /history τα επιστρέφει όλα μαζί. ---
let allHistoryRecords = [];
let currentHistoryPage = 1;
const HISTORY_PAGE_SIZE = 15;
let historySortKey = 'created_at';
let historySortDir = 'desc';
let selectedHistoryIds = new Set();

const historySearchInput = document.getElementById('historySearchInput');
const historyDateInput = document.getElementById('historyDateInput');
historySearchInput.addEventListener('input', () => { currentHistoryPage = 1; renderHistoryTable(); });
historyDateInput.addEventListener('change', () => { currentHistoryPage = 1; renderHistoryTable(); });

function clearHistoryFilters() {
    historySearchInput.value = '';
    historyDateInput.value = '';
    currentHistoryPage = 1;
    renderHistoryTable();
}

function setHistorySort(key) {
    if (historySortKey === key) {
        historySortDir = historySortDir === 'asc' ? 'desc' : 'asc';
    } else {
        historySortKey = key;
        historySortDir = 'desc';
    }
    currentHistoryPage = 1;
    renderHistoryTable();
}

function updateSortIndicators() {
    const iconIds = { confidence: 'sortIconConfidence', created_at: 'sortIconCreated_at' };
    for (const key in iconIds) {
        const el = document.getElementById(iconIds[key]);
        if (!el) continue;
        if (historySortKey === key) {
            el.className = historySortDir === 'asc' ? 'fa-solid fa-sort-up' : 'fa-solid fa-sort-down';
        } else {
            el.className = 'fa-solid fa-sort';
        }
    }
}

function getFilteredHistory() {
    const search = historySearchInput.value.trim().toLowerCase();
    const dateFilter = historyDateInput.value; // 'YYYY-MM-DD'
    const filtered = allHistoryRecords.filter(item => {
        const matchesSearch = !search || item.fruit_name.toLowerCase().includes(search);
        const matchesDate = !dateFilter || (item.created_at && item.created_at.slice(0, 10) === dateFilter);
        return matchesSearch && matchesDate;
    });

    filtered.sort((a, b) => {
        const cmp = historySortKey === 'confidence'
            ? a.confidence - b.confidence
            : new Date(a.created_at) - new Date(b.created_at);
        return historySortDir === 'asc' ? cmp : -cmp;
    });

    return filtered;
}

function renderHistoryTable() {
    updateSortIndicators();
    const tableBody = document.getElementById('historyTableBody');
    const filtered = getFilteredHistory();

    if (filtered.length === 0) {
        const message = allHistoryRecords.length === 0
            ? 'Δεν υπάρχουν καταγεγραμμένες αναλύσεις.'
            : 'Καμία εγγραφή δεν ταιριάζει με τα φίλτρα.';
        tableBody.innerHTML = `
            <tr><td colspan="7">
                <div class="empty-state">
                    <i class="fa-solid fa-box-open"></i>
                    <p>${message}</p>
                </div>
            </td></tr>`;
        renderPagination(0);
        updateSelectAllCheckboxState();
        updateBulkActionBar();
        return;
    }

    const totalPages = Math.max(1, Math.ceil(filtered.length / HISTORY_PAGE_SIZE));
    currentHistoryPage = Math.min(currentHistoryPage, totalPages);
    const start = (currentHistoryPage - 1) * HISTORY_PAGE_SIZE;
    const pageItems = filtered.slice(start, start + HISTORY_PAGE_SIZE);

    tableBody.innerHTML = '';
    pageItems.forEach(item => {
        const dateFormatted = new Date(item.created_at).toLocaleString('el-GR');
        tableBody.innerHTML += `
            <tr id="history-row-${item.id}">
                <td><input type="checkbox" class="history-row-checkbox" data-id="${item.id}" ${selectedHistoryIds.has(item.id) ? 'checked' : ''} onchange="toggleRowSelection(${item.id})"></td>
                <td>#${item.id}</td>
                <td>${item.filename}</td>
                <td><span class="badge">${item.fruit_name}</span></td>
                <td><b>${item.confidence}%</b></td>
                <td style="color: var(--text-muted); font-size: 0.85rem;">${dateFormatted}</td>
                <td>
                    <button class="gradcam-view-btn" onclick="viewHistoryGradcam(${item.id})" title="Προβολή Grad-CAM">
                        <i class="fa-solid fa-eye"></i>
                    </button>
                    <button class="delete-row-btn" onclick="deleteHistoryRecord(${item.id})" title="Διαγραφή εγγραφής">
                        <i class="fa-solid fa-trash-can"></i>
                    </button>
                </td>
            </tr>
        `;
    });

    renderPagination(totalPages);
    updateSelectAllCheckboxState();
    updateBulkActionBar();
}

// --- Μαζική επιλογή/διαγραφή ---
function toggleRowSelection(id) {
    if (selectedHistoryIds.has(id)) selectedHistoryIds.delete(id);
    else selectedHistoryIds.add(id);
    updateSelectAllCheckboxState();
    updateBulkActionBar();
}

function toggleSelectAllOnPage() {
    const checkboxes = document.querySelectorAll('.history-row-checkbox');
    const allSelected = Array.from(checkboxes).every(cb => selectedHistoryIds.has(Number(cb.dataset.id)));
    checkboxes.forEach(cb => {
        const id = Number(cb.dataset.id);
        if (allSelected) selectedHistoryIds.delete(id);
        else selectedHistoryIds.add(id);
    });
    renderHistoryTable();
}

function updateSelectAllCheckboxState() {
    const selectAllCb = document.getElementById('selectAllCheckbox');
    if (!selectAllCb) return;
    const rowCheckboxes = document.querySelectorAll('.history-row-checkbox');
    if (rowCheckboxes.length === 0) {
        selectAllCb.checked = false;
        selectAllCb.indeterminate = false;
        return;
    }
    const selectedCount = Array.from(rowCheckboxes).filter(cb => selectedHistoryIds.has(Number(cb.dataset.id))).length;
    selectAllCb.checked = selectedCount === rowCheckboxes.length;
    selectAllCb.indeterminate = selectedCount > 0 && selectedCount < rowCheckboxes.length;
}

function updateBulkActionBar() {
    const bar = document.getElementById('bulkActionBar');
    const countEl = document.getElementById('bulkSelectedCount');
    if (selectedHistoryIds.size > 0) {
        bar.style.display = 'flex';
        countEl.textContent = `${selectedHistoryIds.size} επιλεγμένες`;
    } else {
        bar.style.display = 'none';
    }
}

async function bulkDeleteSelected() {
    const ids = Array.from(selectedHistoryIds);
    if (ids.length === 0) return;
    const confirmed = await confirmDialog(`Διαγραφή ${ids.length} επιλεγμένων εγγραφών; Η ενέργεια δεν αναιρείται.`);
    if (!confirmed) return;
    try {
        const response = await fetch(`${API_URL}/history/bulk-delete`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ ids }),
        });
        if (!response.ok) throw new Error('Αποτυχία μαζικής διαγραφής.');
        allHistoryRecords = allHistoryRecords.filter(r => !selectedHistoryIds.has(r.id));
        selectedHistoryIds.clear();
        renderHistoryTable();
        showToast(`Διαγράφηκαν ${ids.length} εγγραφές.`, 'success');
    } catch (error) {
        showToast('Δεν ήταν δυνατή η μαζική διαγραφή. Δοκίμασε ξανά.', 'error');
    }
}

// --- Export dropdown (Excel/CSV/JSON) ---
function toggleExportMenu() {
    document.getElementById('exportMenu').classList.toggle('active');
}
document.addEventListener('click', (e) => {
    const dropdown = document.querySelector('.export-dropdown');
    const menu = document.getElementById('exportMenu');
    if (dropdown && menu && !dropdown.contains(e.target)) {
        menu.classList.remove('active');
    }
});

function renderPagination(totalPages) {
    const el = document.getElementById('historyPagination');
    if (totalPages <= 1) { el.innerHTML = ''; return; }
    let html = `<button class="page-btn" ${currentHistoryPage === 1 ? 'disabled' : ''} onclick="goToHistoryPage(${currentHistoryPage - 1})"><i class="fa-solid fa-chevron-left"></i></button>`;
    for (let p = 1; p <= totalPages; p++) {
        html += `<button class="page-btn ${p === currentHistoryPage ? 'active' : ''}" onclick="goToHistoryPage(${p})">${p}</button>`;
    }
    html += `<button class="page-btn" ${currentHistoryPage === totalPages ? 'disabled' : ''} onclick="goToHistoryPage(${currentHistoryPage + 1})"><i class="fa-solid fa-chevron-right"></i></button>`;
    el.innerHTML = html;
}

function goToHistoryPage(page) {
    currentHistoryPage = page;
    renderHistoryTable();
}

async function loadHistory() {
    const tableBody = document.getElementById('historyTableBody');
    try {
        const response = await fetch(`${API_URL}/history`);
        const data = await response.json();
        allHistoryRecords = data;
        renderHistoryTable();
    } catch (error) {
        tableBody.innerHTML = `<tr><td colspan="7" style="text-align: center; color: red;">Αποτυχία φόρτωσης ιστορικού.</td></tr>`;
        showToast('Αποτυχία φόρτωσης ιστορικού.', 'error');
    }
}

async function deleteHistoryRecord(recordId) {
    const confirmed = await confirmDialog(`Διαγραφή της εγγραφής #${recordId}; Η ενέργεια δεν αναιρείται.`);
    if (!confirmed) return;
    try {
        const response = await fetch(`${API_URL}/history/${recordId}`, { method: 'DELETE' });
        if (!response.ok) throw new Error('Αποτυχία διαγραφής.');
        allHistoryRecords = allHistoryRecords.filter(r => r.id !== recordId);
        selectedHistoryIds.delete(recordId);
        renderHistoryTable();
        showToast(`Η εγγραφή #${recordId} διαγράφηκε.`, 'success');
    } catch (error) {
        showToast('Δεν ήταν δυνατή η διαγραφή της εγγραφής. Δοκίμασε ξανά.', 'error');
    }
}

// --- Grad-CAM lightbox (προβολή αποθηκευμένου heatmap από το ιστορικό) ---
const gradcamOverlay = document.getElementById('gradcamOverlay');
const gradcamLightboxBody = document.getElementById('gradcamLightboxBody');
const gradcamLightboxId = document.getElementById('gradcamLightboxId');

function closeGradcamLightbox() {
    gradcamOverlay.classList.remove('active');
    gradcamLightboxBody.innerHTML = '';
}
document.getElementById('gradcamLightboxClose').addEventListener('click', closeGradcamLightbox);
gradcamOverlay.addEventListener('click', (e) => { if (e.target === gradcamOverlay) closeGradcamLightbox(); });

// Esc κλείνει όποιο modal είναι ανοιχτό (Grad-CAM lightbox ή confirm dialog).
// Το confirmCancelBtn.click() είναι ασφαλές ακόμα κι όταν δεν υπάρχει ανοιχτό
// confirm dialog - τότε δεν έχει κανέναν attached listener, άρα δεν κάνει τίποτα.
document.addEventListener('keydown', (e) => {
    if (e.key !== 'Escape') return;
    if (gradcamOverlay.classList.contains('active')) {
        closeGradcamLightbox();
    } else if (confirmOverlay.classList.contains('active')) {
        confirmCancelBtn.click();
    }
});

async function viewHistoryGradcam(recordId) {
    gradcamLightboxId.textContent = `#${recordId}`;
    gradcamLightboxBody.innerHTML = `
        <div class="gradcam-lightbox-empty">
            <div class="spinner" style="display: block;"></div>
        </div>`;
    gradcamOverlay.classList.add('active');

    try {
        const response = await fetch(`${API_URL}/history/${recordId}/gradcam`);
        if (!response.ok) throw new Error('not found');
        const data = await response.json();
        gradcamLightboxBody.innerHTML = `<img src="${data.gradcam_image}" alt="Grad-CAM heatmap">`;
    } catch (error) {
        gradcamLightboxBody.innerHTML = `
            <div class="gradcam-lightbox-empty">
                <i class="fa-solid fa-image"></i>
                <p>Δεν υπάρχει αποθηκευμένο Grad-CAM για αυτή την εγγραφή.</p>
            </div>`;
    }
}

loadHistory();
