import numpy as np
from utils.HRV_Functions import *


def draw1(x, xminn, fs, dat, plot_fig, graphcanvas, fig, rt, ramp, loaded_ann, labelled_flag, trt,
          tramp, disp_length, plot_pred, plot_ann, fig2, plot_fig2, rri_canvas):
    """
    Draw 1
    """
    if x is not None:
        xmaxx = int(xminn + disp_length * fs)
        yminn = np.min(dat[int(xminn):xmaxx]) - 0.1
        ymaxx = np.max(dat[int(xminn):xmaxx]) + 0.1

        # Top Figure
        plot_fig.clear()
        plot_fig.plot(x / fs, dat, color='k', linewidth=1)
        if labelled_flag & plot_pred:
            plot_fig.plot(rt / fs, ramp, 'b*', linewidth=1, markersize=7)
        if loaded_ann & plot_ann:
            plot_fig.plot(trt / fs, tramp, 'ro', linewidth=1, markersize=5, fillstyle='none')
        plot_fig.axis([xminn / fs, xmaxx / fs, yminn, ymaxx])  # ([xminn,xmaxx,yminn,ymaxx])
        plot_fig.set_xlabel('Time (sec)')
        plot_fig.set_ylabel('ECG Amplitude (mV)')
        fig.tight_layout()
        graphcanvas.draw()

        draw2(xmaxx, rt, trt, fig2, plot_fig2, rri_canvas, xminn, plot_ann, plot_pred, fs)


def draw2(xmaxx: int, rt, trt, fig2, plot_fig2, rri_canvas, xminn: int, plot_ann, plot_pred, fs):

    if plot_ann ^ plot_pred:
        if plot_pred:
            plotx = rt
        else:
            plotx = trt

        plotx = np.reshape(plotx, (len(plotx), 1))

        x2_max = np.max(plotx) / fs
        srt = np.size(plotx)

        y_diff = (np.diff(plotx[:, 0]) / fs) * 1000

        pl = xminn / fs
        pl2 = xmaxx / fs

        y_minn = np.min(y_diff) - 10
        y_maxx = np.max(y_diff) + 10

        plot_fig2.clear()
        x2 = plotx[1:srt] / fs
        x2 = np.reshape(x2, (len(x2),))

        if plot_pred:
            plot_fig2.plot(x2, y_diff, 'b*', label='Predicted beats')
        else:
            plot_fig2.plot(x2, y_diff, 'ro', label='Annotated beats', markersize=5, fillstyle='none')

        plot_fig2.plot([pl + 0.05, pl + 0.05], [y_minn, y_maxx], 'k')
        plot_fig2.plot([pl2, pl2], [y_minn, y_maxx], 'k')
        plot_fig2.axis([0, x2_max, y_minn, y_maxx])
        plot_fig2.set_xlabel('Time (sec)')
        plot_fig2.set_ylabel('R-R Interval (ms)')
        plot_fig2.fill_between(x2, y_minn, y_maxx, where=x2 <= pl2, facecolor='gainsboro')
        plot_fig2.fill_between(x2, y_minn, y_maxx, where=x2 <= pl, facecolor='white')
        plot_fig2.legend()
        fig2.tight_layout()
        rri_canvas.draw()

    elif plot_ann & plot_pred:
        plot_fig2.clear()
        y_minn = np.zeros(2)
        y_maxx = np.zeros(2)
        x2_maxx = np.zeros(2)

        #        pl = xminn/Fs#np.argmin(np.abs(rt-xminn))
        #        pl2 = xmaxx/Fs# np.argmin(np.abs(rt-xmaxx))
        #
        #        for i in range (0, (sRt-1)):
        #            y_diff[i] = plotx[i+1] - plotx[i]
        #
        #        y_diff = (y_diff/Fs)*1000   #Converts from sample number to millseconds
        #        x2 = range(0, (sRt-1))

        #
        for N in range(2):
            if N == 0:
                plotx = rt
            else:
                plotx = trt
            srt = np.size(plotx)
            y_diff = (np.diff(plotx[:, 0]) / fs) * 1000
            y_minn[N] = np.min(y_diff) - 10
            y_maxx[N] = np.max(y_diff) + 10
            x2_maxx[N] = np.max(plotx) / fs

            if N == 0:
                plot_fig2.plot(plotx[1:srt] / fs, y_diff, 'b*', label='Predicted beats')
            else:
                plot_fig2.plot(plotx[1:srt] / fs, y_diff, 'ro', label='Annotated beats', markersize=5, fillstyle='none')

        plot_fig2.set_xlabel('Time (sec)')
        plot_fig2.set_ylabel('R-R Interval (ms)')
        if y_minn[0] < y_minn[1]:
            plot_min = y_minn[0]
        else:
            plot_min = y_minn[1]
        if y_maxx[0] > y_maxx[1]:
            plot_max = y_maxx[0]
        else:
            plot_max = y_maxx[1]
        if x2_maxx[0] < x2_maxx[1]:
            plot_max_x2 = x2_maxx[0]
        else:
            plot_max_x2 = x2_maxx[1]
        plot_fig2.plot([(xminn / fs) + 0.5, (xminn / fs) + 0.5], [y_minn, y_maxx], 'k')
        plot_fig2.plot([(xmaxx / fs), (xmaxx / fs)], [y_minn, y_maxx], 'k')
        plot_fig2.axis([0, plot_max_x2, plot_min, plot_max])
        plot_fig2.legend()
        fig2.tight_layout()
        rri_canvas.draw()


def draw3(rt, fig2, plot_fig2, rricanvas, fs: int):
    """
    Draw Number 3
    """
    plotx = np.reshape(rt, (len(rt), 1))
    x2_max = np.max(plotx) / fs
    sRt = np.size(plotx)
    y_diff = (np.diff(plotx[:, 0]) / fs) * 1000
    y_minn = np.min(y_diff) - 10
    y_maxx = np.max(y_diff) + 10
    plot_fig2.clear()
    x2 = plotx[1:sRt] / fs
    x2 = np.reshape(x2, (len(x2),))
    plot_fig2.plot(x2, y_diff, 'b*', label='R-peaks')
    plot_fig2.axis([0, x2_max, y_minn, y_maxx])
    plot_fig2.set_xlabel('Time (sec)')
    plot_fig2.set_ylabel('R-R Interval (ms)')
    plot_fig2.legend()
    fig2.tight_layout()
    rricanvas.draw()


def DFA_plot(subplot_, min_: int = 4, max_: int = 64, inc_: int = 1, cop_: int = 12):
    """

    :param subplot_:
    :param min_:
    :param max_:
    :param inc_:
    :param cop_:
    """
    global draw_figure
    global R_t
    global graphCanvas2
    Rpeakss = np.reshape(R_t, (len(R_t),))
    Rpeak_input = Rpeakss / Fs
    RRI = np.diff(Rpeak_input)
    (x_vals1, y_vals1, y_new1, x_vals2, y_vals2, y_new2, a1, a2) = DFA_fig(RRI, min_box=min_, max_box=max_, inc=inc_,
                                                                           cop=cop_, decim=3)
    subplot_.clear()
    subplot_.plot(x_vals1, y_vals1, '*')
    subplot_.plot(x_vals1, y_new1, 'r--')
    txt = "\u03B1$_1$ = {:.3f}".format(a1)
    subplot_.text(0.8, np.min(y_vals1) + 0.1, txt)
    subplot_.plot(x_vals2, y_vals2, '*')
    subplot_.plot(x_vals2, y_new2, 'k--')
    txt = "\u03B1$_2$ = {:.3f}".format(a2)
    subplot_.text(1.5, np.mean(y_vals2) * 1.1, txt)
    subplot_.set_title('Detrended Fluctuation Analysis of the RRI series')
    subplot_.set_xlabel('log$_{10}$ n (beats)')
    subplot_.set_ylabel('log$_{10}$ F(n) (beats)')
    draw_figure.tight_layout()
    graphCanvas2.draw()


def freq_plot(METHOD, DECIMALS, subplot_, m=1200, o=600, btval=10, omax=500, Ord=100):
    """
    :param METHOD:
    :param DECIMALS:
    :param subplot_:
    :param m:
    :param o:
    :param btval:
    :param omax:
    :param Ord:
    """
    global draw_figure
    global R_t
    global graphCanvas2
    Rpeakss = np.reshape(R_t, (len(R_t),))
    Rpeak_input = Rpeakss / Fs
    fig_holder = Figure(dpi=150)
    f, P2 = Freq_Analysis_fig(Rpeak_input, meth=METHOD, decim=DECIMALS, Fig=fig_holder, M=m, O=o, BTval=btval,
                              omega_max=omax, order=Ord)
    title_list = ['Welch\'s', 'Blackman-Tukey\'s', 'LombScargle\'s', 'Auto Regression']
    subplot_.clear()
    subplot_.xaxis.tick_bottom()
    subplot_.plot(f, P2, 'black', linewidth=0.75, label='PSD')  # ,
    subplot_.set_xlim(0, 0.5)
    subplot_.set_ylim(ymin=0)
    subplot_.set_title("PSD Estimation using " + title_list[METHOD - 1] + " method")
    subplot_.set_xlabel("Frequency (Hz)")
    subplot_.set_ylabel("Power Spectral Density (s$^{2}$/Hz)", labelpad=10)
    subplot_.fill_between(f, 0, P2, where=f <= 0.4, facecolor='red', label='HF')
    subplot_.fill_between(f, 0, P2, where=f <= 0.15, facecolor='blue', label='LF')
    subplot_.fill_between(f, 0, P2, where=f < 0.04, facecolor='green', label='VLF')
    subplot_.legend()
    draw_figure.tight_layout()
    graphCanvas2.draw()


def Poincare_plot(subplot_):
    """

    :param subplot_:
    """
    global draw_figure
    global R_t
    global graphCanvas2
    Rpeakss = np.reshape(R_t, (len(R_t),))
    Rpeak_input = Rpeakss / Fs
    RRI = np.diff(Rpeak_input)
    sd1, sd2, c1, c2 = Poincare(RRI, decim=3)
    lenx = np.size(RRI)
    xp = RRI[0:lenx - 1]
    yp = RRI[1:lenx]
    A = sd2 * np.cos(np.pi / 4)
    B = sd1 * np.sin(np.pi / 4)
    ellipse = patch.Ellipse((c1, c2), sd2 * 2, sd1 * 2, 45, facecolor="none", edgecolor="b", linewidth=2, zorder=5)
    subplot_.clear()
    subplot_.plot(xp, yp, 'ko', markersize=3, zorder=0)  # ,'MarkerFaceColor', 'k', 'MarkerSize',4)
    subplot_.add_patch(ellipse)
    subplot_.set_title('Poincare Plot')
    subplot_.set_xlabel('RRI$_{n}$ (s)')
    subplot_.set_ylabel('RRI$_{n+1}$ (s)')
    subplot_.plot([c1, c1 + A], [c2, c2 + A], 'm', label="SD1", zorder=10)
    subplot_.plot([c1 - 4 * A, c1 + 4 * A], [c2 - 4 * A, c2 + 4 * A], 'b', dashes=[6, 2])
    subplot_.plot([c1, c1 - B], [c2, c2 + B], 'c', label="SD2", zorder=10)
    subplot_.plot([c1 + B * 4, c1 - B * 4], [c2 - 4 * B, c2 + 4 * B], 'b', dashes=[4, 2, 10, 2])
    subplot_.legend()
    draw_figure.tight_layout()
    graphCanvas2.draw()


def RQA_plott(subplot_, canvas, fs, Mval=10, Lval=1):
    """

    :param subplot_:
    :param Mval:
    :param Lval:
    """
    rpeakss = np.reshape(R_t, (len(R_t),))
    Rpeak_input = rpeakss / fs
    RRI = np.diff(Rpeak_input)
    Matrix, N = RQA_matrix(RRI, m=Mval, L=Lval)

    xplot = np.zeros((N, N))
    yplot = np.zeros((N, N))
    subplot_.clear()
    for i in range(0, len(Matrix)):
        yplot[:, i] = np.arange(1, len(Matrix) + 1) * Matrix[:, i]
        xplot[:, i] = np.ones(len(Matrix)) * (i + 1)
    subplot_.scatter(xplot, yplot, c='k', s=0.5)
    subplot_.set_title('Recurrence Plot')
    subplot_.set_xlabel('Heart beat (sample number)')
    subplot_.set_ylabel('Heart beat (sample number)')
    draw_figure.tight_layout()
    canvas.draw()
