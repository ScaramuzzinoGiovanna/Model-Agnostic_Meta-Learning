import matplotlib.pyplot as plt


def plot_regression(array, x_tab, y_tab, x_test, y_test, K, i, name, folder_plot):
    # plot
    ground_truth = plt.plot(x_tab, y_tab, '')[0]
    K_points = plt.plot(x_test, y_test, '^')[0]
    plots = [K_points, ground_truth]
    legend = ['K', 'ground_truth']
    for n, y_pred in array:
        if n == 0:
            cur, = plt.plot(x_tab, y_pred[:, 0], '-.')
            n = 'pre-update'
            legend.append(n)
        else:
            cur, = plt.plot(x_tab, y_pred[:, 0], '--')
            legend.append(f'{n} grad steps')
        plots.append(cur)
    plt.legend(plots, legend)
    plt.ylim(-5, 5)
    plt.xlim(-6, 6)
    plt.title('{} K={}'.format(name, K))
    plt.savefig(folder_plot + '/' + name + '_K' + str(K) + '_test_' + str(i) + '.png')
    plt.show()


def plot_learning_curve(mse_maml, mse_pretrained, K, i, folder_plot):
    # plot learning curve at meta-testing time
    maml = plt.plot(range(len(mse_maml)), mse_maml)[0]
    pre = plt.plot(range(len(mse_pretrained)), mse_pretrained)[0]
    plt.legend([maml, pre], ['MAML', 'Pretrained'])
    plt.title('k-shot regression K={}'.format(K))
    plt.xlabel('number of gradient steps')
    plt.ylabel('mean squared error')
    plt.savefig(folder_plot + '/' + 'mse_match_k{}_test_{}.png'.format(K, i))
    plt.show()


def plot_avgLosses(losses, folder_plots, name_model, K):
    plt.plot(losses)
    plt.title('pretrained_loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title(name_model + 'K={} Avg loss'.format(K))
    plt.savefig(folder_plots + '/' + name_model + '_K{}_loss.png'.format(K))
    plt.show()
