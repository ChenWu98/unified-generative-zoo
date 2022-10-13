import logging


def statistics2str(statistics):
    return str({k: f'{v:.6g}' for k, v in statistics.items()})


def report_statistics(s, t, statistics):
    logging.info(f'[(s, r): ({s:.6g}, {t:.6g})] [{statistics2str(statistics)}]')
