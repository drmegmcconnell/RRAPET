"""
Stylising the Program

"""
import numpy as np
from tkinter import font


def headerStyles(preferences):
    """
    Header Styles
    """
    cust_text = font.Font(family=preferences[0], size=int(preferences[1]))
    cust_subheader = font.Font(family=preferences[0], size=(int(preferences[1]) + 2), weight='bold')
    cust_subheadernb = font.Font(family=preferences[0], size=(int(preferences[1]) + 2))
    cust_header = font.Font(family=preferences[0], size=(int(preferences[1]) + 4))
    return cust_text, cust_subheader, cust_subheadernb, cust_header


def draw1(x, xminn, Fs, dat, plot_fig, graphCanvas, fig, R_t, R_amp, loaded_ann, labelled_flag, True_R_t,
          True_R_amp, disp_length, plot_pred, plot_ann, fig2, plot_fig2, RR_interval_Canvas):
    """
    Draw 1
    """

    xmaxx = int(xminn + disp_length * Fs)
    yminn = np.min(dat[int(xminn):xmaxx]) - 0.1
    ymaxx = np.max(dat[int(xminn):xmaxx]) + 0.1

    # Top Figure
    plot_fig.clear()
    plot_fig.plot(x / Fs, dat, color='k', linewidth=1)
    if labelled_flag & plot_pred:
        plot_fig.plot(R_t / Fs, R_amp, 'b*', linewidth=1, markersize=7)
    if loaded_ann & plot_ann:
        plot_fig.plot(True_R_t / Fs, True_R_amp, 'ro', linewidth=1, markersize=5, fillstyle='none')
    plot_fig.axis([xminn / Fs, xmaxx / Fs, yminn, ymaxx])  # ([xminn,xmaxx,yminn,ymaxx])
    plot_fig.set_xlabel('Time (sec)')
    plot_fig.set_ylabel('ECG Amplitude (mV)')
    fig.tight_layout()
    graphCanvas.draw()

    draw2(xmaxx, R_t, R_amp, True_R_t, True_R_amp, dat, x, fig2, plot_fig2, RR_interval_Canvas, labelled_flag,
          xminn, plot_ann, plot_pred, Fs)


def draw2(xmaxx, R_t, R_amp, True_R_t, True_R_amp, dat, x, fig2, plot_fig2, RR_interval_Canvas, labelled_flag,
          xminn, plot_ann, plot_pred, Fs):
    """
    Draw 2
    """

    if plot_ann ^ plot_pred:
        if plot_pred:
            plotx = R_t
        else:
            plotx = True_R_t

        plotx = np.reshape(plotx, (len(plotx), 1))

        x2_max = np.max(plotx) / Fs
        sRt = np.size(plotx)

        y_diff = (np.diff(plotx[:, 0]) / Fs) * 1000

        pl = xminn / Fs
        pl2 = xmaxx / Fs

        y_minn = np.min(y_diff) - 10
        y_maxx = np.max(y_diff) + 10

        plot_fig2.clear()
        x2 = plotx[1:sRt] / Fs
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
        RR_interval_Canvas.draw()

    elif plot_ann & plot_pred:
        plot_fig2.clear()
        y_minn = np.zeros(2)
        y_maxx = np.zeros(2)
        x2_maxx = np.zeros(2)

        #        pl = xminn/Fs#np.argmin(np.abs(R_t-xminn))
        #        pl2 = xmaxx/Fs# np.argmin(np.abs(R_t-xmaxx))
        #
        #        for i in range (0, (sRt-1)):
        #            y_diff[i] = plotx[i+1] - plotx[i]
        #
        #        y_diff = (y_diff/Fs)*1000   #Converts from sample number to millseconds
        #        x2 = range(0, (sRt-1))

        #
        for N in range(2):
            if N == 0:
                plotx = R_t
            else:
                plotx = True_R_t
            sRt = np.size(plotx)
            y_diff = (np.diff(plotx[:, 0]) / Fs) * 1000
            y_minn[N] = np.min(y_diff) - 10
            y_maxx[N] = np.max(y_diff) + 10
            x2_maxx[N] = np.max(plotx) / Fs

            if N == 0:
                plot_fig2.plot(plotx[1:sRt] / Fs, y_diff, 'b*', label='Predicted beats')
            else:
                plot_fig2.plot(plotx[1:sRt] / Fs, y_diff, 'ro', label='Annotated beats', markersize=5, fillstyle='none')

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
        plot_fig2.plot([(xminn / Fs) + 0.5, (xminn / Fs) + 0.5], [y_minn, y_maxx], 'k')
        plot_fig2.plot([(xmaxx / Fs), (xmaxx / Fs)], [y_minn, y_maxx], 'k')
        plot_fig2.axis([0, plot_max_x2, plot_min, plot_max])
        plot_fig2.legend()
        fig2.tight_layout()
        RR_interval_Canvas.draw()
