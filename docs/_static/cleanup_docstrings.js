// Remove orphaned RST code-block directive text that mkdocstrings doesn't fully process
document.addEventListener('DOMContentLoaded', function () {
    document.querySelectorAll('.doc p').forEach(function (p) {
        if (/^\s*\.\. code-block::/.test(p.textContent)) {
            p.remove();
        }
    });
});
