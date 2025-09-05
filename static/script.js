// Menunggu seluruh konten halaman dimuat sebelum menjalankan skrip
document.addEventListener("DOMContentLoaded", () => {
  // --- LOGIKA UNTUK DARK MODE (DIPERBAIKI) ---
  const themeToggleBtn = document.getElementById("theme-toggle");
  const currentTheme = localStorage.getItem("theme");

  // Saat halaman dimuat, terapkan tema yang tersimpan
  if (currentTheme === "dark") {
    document.body.classList.add("dark-mode");
  }

  // Tambahkan event listener untuk tombol jika tombol ada
  if (themeToggleBtn) {
    themeToggleBtn.addEventListener("click", () => {
      // Toggle kelas 'dark-mode' dari body
      document.body.classList.toggle("dark-mode");

      // Tentukan tema berdasarkan status kelas
      let theme = document.body.classList.contains("dark-mode")
        ? "dark"
        : "light";

      // Simpan pilihan tema ke localStorage
      localStorage.setItem("theme", theme);
    });
  }
  // --- AKHIR LOGIKA DARK MODE ---

  // --- LOGIKA UNTUK MENU HAMBURGER ---
  const hamburger = document.getElementById("hamburger-menu");
  const navMenu = document.getElementById("navigation-menu");

  if (hamburger && navMenu) {
    hamburger.addEventListener("click", () => {
      navMenu.classList.toggle("is-active");
    });
  }

  // --- LOGIKA UNTUK FORM FACE SWAP ---
  const swapForm = document.getElementById("swap-form");

  if (swapForm) {
    const sourceInput = document.getElementById("source-image");
    const targetInput = document.getElementById("target-image");
    const sourcePreview = document.getElementById("source-preview");
    const targetPreview = document.getElementById("target-preview");
    const resultArea = document.getElementById("result-area");
    const resultImage = document.getElementById("result-image");
    const loader = document.getElementById("loader");
    const submitBtn = document.getElementById("submit-btn");
    const downloadBtn = document.getElementById("download-btn");
    const shareSection = document.getElementById("share-section");

    function setupImagePreview(input, previewElement) {
      input.addEventListener("change", () => {
        if (input.files && input.files[0]) {
          const reader = new FileReader();
          reader.onload = (e) => {
            previewElement.src = e.target.result;
            previewElement.style.display = "block";
          };
          reader.readAsDataURL(input.files[0]);
        }
      });
    }

    setupImagePreview(sourceInput, sourcePreview);
    setupImagePreview(targetInput, targetPreview);

    swapForm.addEventListener("submit", async (e) => {
      e.preventDefault();
      resultArea.classList.remove("hidden");
      loader.style.display = "block";
      resultImage.style.display = "none";
      downloadBtn.classList.add("hidden");
      shareSection.classList.add("hidden"); // Sembunyikan share section
      submitBtn.disabled = true;
      submitBtn.innerHTML =
        '<i class="fas fa-spinner fa-spin"></i> Memproses...';

      const formData = new FormData();
      formData.append("source_image", sourceInput.files[0]);
      formData.append("target_image", targetInput.files[0]);

      try {
        const response = await fetch("/swap", {
          method: "POST",
          body: formData,
        });
        const data = await response.json();
        if (response.ok) {
          resultImage.src = data.result_image;
          resultImage.style.display = "block";
          downloadBtn.href = data.result_image;
          downloadBtn.classList.remove("hidden");

          // Tampilkan share section setelah hasil berhasil
          shareSection.classList.remove("hidden");
        } else {
          // Sembunyikan share section jika error
          shareSection.classList.add("hidden");
          Swal.fire({
            icon: "error",
            title: "Oops... Terjadi Kesalahan",
            text: data.error,
            confirmButtonColor: "#d33",
            confirmButtonText: "Mengerti",
          });
        }
      } catch (error) {
        shareSection.classList.add("hidden");
        Swal.fire({
          icon: "error",
          title: "Koneksi Gagal",
          text: "Tidak dapat terhubung ke server.",
          confirmButtonColor: "#d33",
          confirmButtonText: "Mengerti",
        });
        console.error("Fetch error:", error);
      } finally {
        loader.style.display = "none";
        submitBtn.disabled = false;
        submitBtn.innerHTML =
          '<i class="fas fa-exchange-alt"></i> Tukar Wajah!';
      }
    });
  }

  // === LOGIKA BARU UNTUK KONFIRMASI HAPUS ===
  const deleteForms = document.querySelectorAll(".delete-form");
  deleteForms.forEach((form) => {
    form.addEventListener("submit", function (e) {
      e.preventDefault(); // Mencegah form langsung dikirim

      Swal.fire({
        title: "Anda Yakin?",
        text: "Gambar ini akan dihapus secara permanen!",
        icon: "warning",
        showCancelButton: true,
        confirmButtonColor: "#3085d6",
        cancelButtonColor: "#d33",
        confirmButtonText: "Ya, hapus!",
        cancelButtonText: "Batal",
      }).then((result) => {
        if (result.isConfirmed) {
          // Jika pengguna mengklik "Ya, hapus!", maka kirim form
          form.submit();
        }
      });
    });
  });

  // --- LOGIKA UNTUK TEXT SWAPPER ---
  const textSwapForm = document.getElementById("text-swap-form");

  if (textSwapForm) {
    const textImageInput = document.getElementById("text-image");
    const textPreview = document.getElementById("text-preview");
    const textResultArea = document.getElementById("text-result-area");
    const textResultImage = document.getElementById("text-result-image");
    const textLoader = document.getElementById("text-loader");
    const textSubmitBtn = document.getElementById("text-submit-btn");
    const textDownloadBtn = document.getElementById("text-download-btn");
    const textShareSection = document.getElementById("text-share-section");

    // Preview gambar
    textImageInput.addEventListener("change", () => {
      if (textImageInput.files && textImageInput.files[0]) {
        const reader = new FileReader();
        reader.onload = (e) => {
          textPreview.src = e.target.result;
          textPreview.style.display = "block";
        };
        reader.readAsDataURL(textImageInput.files[0]);
      }
    });

    // Submit form
    textSwapForm.addEventListener("submit", async (e) => {
      e.preventDefault();
      textResultArea.classList.remove("hidden");
      textLoader.style.display = "block";
      textResultImage.style.display = "none";
      textDownloadBtn.classList.add("hidden");
      textShareSection.classList.add("hidden");
      textSubmitBtn.disabled = true;
      textSubmitBtn.innerHTML =
        '<i class="fas fa-spinner fa-spin"></i> Memproses...';

      const formData = new FormData();
      formData.append("image", textImageInput.files[0]);
      formData.append(
        "original_text",
        document.getElementById("original-text").value
      );
      formData.append("new_text", document.getElementById("new-text").value);

      try {
        const response = await fetch("/text-swap-process", {
          method: "POST",
          body: formData,
        });
        const data = await response.json();

        if (response.ok) {
          textResultImage.src = data.result_image;
          textResultImage.style.display = "block";
          textDownloadBtn.href = data.result_image;
          textDownloadBtn.classList.remove("hidden");
          textShareSection.classList.remove("hidden");
        } else {
          textShareSection.classList.add("hidden");
          Swal.fire({
            icon: "error",
            title: "Oops... Terjadi Kesalahan",
            text: data.error,
            confirmButtonColor: "#d33",
            confirmButtonText: "Mengerti",
          });
        }
      } catch (error) {
        textShareSection.classList.add("hidden");
        Swal.fire({
          icon: "error",
          title: "Koneksi Gagal",
          text: "Tidak dapat terhubung ke server.",
          confirmButtonColor: "#d33",
          confirmButtonText: "Mengerti",
        });
        console.error("Fetch error:", error);
      } finally {
        textLoader.style.display = "none";
        textSubmitBtn.disabled = false;
        textSubmitBtn.innerHTML =
          '<i class="fas fa-exchange-alt"></i> Ganti Text!';
      }
    });
  }
  // ==========================================
});

// ===============================================
// --- FUNGSI-FUNGSI UNTUK SHARE KE MEDIA SOSIAL ---
// ===============================================

// Fungsi untuk share ke Facebook
function shareToFacebook() {
  const shareUrl = encodeURIComponent(window.location.href);
  const shareText = encodeURIComponent(
    "Lihat hasil AI Face Swap saya! Dibuat dengan AI Swapper 🤖✨"
  );
  const facebookUrl = `https://www.facebook.com/sharer/sharer.php?u=${shareUrl}&quote=${shareText}`;
  window.open(
    facebookUrl,
    "_blank",
    "width=600,height=400,scrollbars=yes,resizable=yes"
  );
}

// Fungsi untuk share ke Twitter
function shareToTwitter() {
  const shareUrl = encodeURIComponent(window.location.href);
  const shareText = encodeURIComponent(
    "Lihat hasil AI Face Swap saya! 🤖✨ Dibuat dengan AI Swapper #AIFaceSwap #AI #FaceSwap"
  );
  const twitterUrl = `https://twitter.com/intent/tweet?text=${shareText}&url=${shareUrl}`;
  window.open(
    twitterUrl,
    "_blank",
    "width=600,height=400,scrollbars=yes,resizable=yes"
  );
}

// Fungsi untuk share ke WhatsApp
function shareToWhatsApp() {
  const shareText = encodeURIComponent(
    `Lihat hasil AI Face Swap saya! 🤖✨ Dibuat dengan AI Swapper\n\n${window.location.href}`
  );
  const whatsappUrl = `https://wa.me/?text=${shareText}`;
  window.open(whatsappUrl, "_blank");
}

// Fungsi untuk share ke Instagram (memberikan instruksi)
function shareToInstagram() {
  Swal.fire({
    title: "Share ke Instagram",
    html: `
            <div style="text-align: left;">
                <p style="margin-bottom: 15px;"><strong>Instagram tidak mendukung share langsung dari web.</strong></p>
                <p style="margin-bottom: 10px;">Untuk share ke Instagram, silakan:</p>
                <ol style="margin: 15px 0; padding-left: 20px;">
                    <li style="margin-bottom: 8px;">Klik tombol <strong>"Download Gambar"</strong></li>
                    <li style="margin-bottom: 8px;">Buka aplikasi <strong>Instagram</strong> di HP Anda</li>
                    <li style="margin-bottom: 8px;">Pilih <strong>"+"</strong> untuk upload gambar</li>
                    <li style="margin-bottom: 8px;">Pilih gambar yang sudah didownload</li>
                    <li>Tambahkan caption dan posting! ✨</li>
                </ol>
                <p style="margin-top: 15px; font-size: 14px; color: #666;">💡 <em>Tip: Tambahkan hashtag #AIFaceSwap untuk lebih viral!</em></p>
            </div>
        `,
    icon: "info",
    confirmButtonText: '<i class="fas fa-download"></i> Download Gambar',
    confirmButtonColor: "#1a73e8",
    showCancelButton: true,
    cancelButtonText: "Tutup",
    cancelButtonColor: "#6c757d",
    width: "500px",
  }).then((result) => {
    if (result.isConfirmed) {
      // Trigger download
      const downloadBtn = document.getElementById("download-btn");
      if (downloadBtn) {
        downloadBtn.click();

        // Tampilkan notifikasi download
        setTimeout(() => {
          Swal.fire({
            title: "Download Dimulai!",
            text: "Gambar sedang didownload. Sekarang buka Instagram untuk upload! 📱",
            icon: "success",
            timer: 3000,
            showConfirmButton: false,
          });
        }, 500);
      }
    }
  });
}

// Fungsi untuk copy link gambar
function copyImageLink() {
  const linkToCopy = window.location.href;

  // Coba gunakan modern clipboard API
  if (navigator.clipboard && window.isSecureContext) {
    navigator.clipboard
      .writeText(linkToCopy)
      .then(() => {
        showCopySuccess();
      })
      .catch(() => {
        fallbackCopyTextToClipboard(linkToCopy);
      });
  } else {
    // Fallback untuk browser lama
    fallbackCopyTextToClipboard(linkToCopy);
  }
}

// Fungsi fallback untuk copy text (untuk browser lama)
function fallbackCopyTextToClipboard(text) {
  const textArea = document.createElement("textarea");
  textArea.value = text;

  // Avoid scrolling to bottom
  textArea.style.top = "0";
  textArea.style.left = "0";
  textArea.style.position = "fixed";

  document.body.appendChild(textArea);
  textArea.focus();
  textArea.select();

  try {
    const successful = document.execCommand("copy");
    if (successful) {
      showCopySuccess();
    } else {
      showCopyFallback(text);
    }
  } catch (err) {
    console.error("Fallback: Oops, unable to copy", err);
    showCopyFallback(text);
  }

  document.body.removeChild(textArea);
}

// Fungsi untuk menampilkan notifikasi copy berhasil
function showCopySuccess() {
  const copyBtn = document.getElementById("copy-link-btn");
  if (copyBtn) {
    const originalHTML = copyBtn.innerHTML;

    // Ubah tampilan tombol sementara
    copyBtn.innerHTML = '<i class="fas fa-check"></i> Copied!';
    copyBtn.classList.add("copy-success");

    // Kembalikan ke tampilan semula setelah 2 detik
    setTimeout(() => {
      copyBtn.innerHTML = originalHTML;
      copyBtn.classList.remove("copy-success");
    }, 2000);
  }

  // Tampilkan notifikasi sukses
  Swal.fire({
    title: "Link Berhasil Disalin!",
    text: "Link sudah tersalin ke clipboard. Sekarang bisa di-paste di mana saja! 📋",
    icon: "success",
    timer: 2000,
    showConfirmButton: false,
    toast: true,
    position: "top-end",
    timerProgressBar: true,
  });
}

// Fungsi fallback jika clipboard tidak bisa diakses
function showCopyFallback(text) {
  Swal.fire({
    title: "Copy Link Secara Manual",
    html: `
            <p style="margin-bottom: 15px;">Silakan copy link di bawah ini secara manual:</p>
            <input type="text" value="${text}" readonly 
                   style="width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 5px; font-size: 12px; background: #f9f9f9;"
                   onclick="this.select()" />
            <p style="margin-top: 15px; font-size: 14px; color: #666;">
                💡 <em>Tip: Klik pada link di atas untuk select semua, lalu tekan Ctrl+C (Windows) atau Cmd+C (Mac)</em>
            </p>
        `,
    icon: "info",
    confirmButtonText: "Mengerti",
    confirmButtonColor: "#1a73e8",
    width: "500px",
  });
}
