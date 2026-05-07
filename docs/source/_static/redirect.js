// Redirect visitors from RTD to docs.pathsim.org after a brief delay.
// The banner is shown immediately; redirect fires after 3 seconds
// so users understand what's happening. Click the link to go immediately.
(function () {
    if (window.location.hostname.indexOf('readthedocs') === -1) return;
    var target = 'https://docs.pathsim.org';
    setTimeout(function () { window.location.replace(target); }, 3000);
})();
