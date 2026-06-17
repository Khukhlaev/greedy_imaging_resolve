from utils.utilities import get_source_date
from casatasks import importuvfits, mstransform, flagdata
from casatools import msmetadata
import shutil
import argparse
import os


parser = argparse.ArgumentParser(description='Transfrom uvf data to ms format.')
parser.add_argument('--data_file', type=str, help='path to the data file.')
args = parser.parse_args()

filename = os.path.basename(args.data_file)
source, _ = get_source_date(args.data_file)
ms_path = f"./ms_data/{source}/{filename}.ms"
tmp_ms_path = f"./ms_data/tmp/{filename}.ms"

if not os.path.exists(ms_path):
    importuvfits(args.data_file, vis=tmp_ms_path)
    print("Sucessfully transformed to ms format")

    # Reading metadata to determine which SPWs have unflagged data
    msmd = msmetadata()
    msmd.open(tmp_ms_path)

    good_spws = []
    for spw in range(msmd.nspw()):
        s = flagdata(vis=tmp_ms_path, spw=str(spw), mode="summary")
        total = s.get("total", 0)
        flagged = s.get("flagged", 0)

        # keep only SPWs that still have some unflagged data
        if total > 0 and flagged < total:
            good_spws.append(spw)

    nchan = [msmd.nchan(spw) for spw in good_spws]
    msmd.done()

    if not good_spws:
        print(f"All SPWs are fully flagged for {args.data_file}, skipping.")
        raise RuntimeError("All SPWs are fully flagged.")

    if nchan == [1] and len(good_spws) == 1: # If there's only one SPW with one channel, just copy it to the final location without averaging
        shutil.copytree(tmp_ms_path, ms_path)
    
    else:

        if len(set(nchan)) != 1: # All SPWs must have the same number of channels to be combined
            print(f"SPWs have different channel counts for {args.data_file}, cannot combine: {nchan}")
            raise ValueError(
                f"combinespws=True requires equal channel counts, got {nchan}. "
            )

        # Combine SPWs, then average the full combined spectral axis into 1 channel
        mstransform(
            vis=tmp_ms_path,
            outputvis=ms_path,
            datacolumn="DATA",
            spw=",".join(str(spw) for spw in good_spws),
            combinespws=True,
            chanaverage=True,
            chanbin=sum(nchan),       # one output channel from the full combined axis
        )


    # finally, remove the temporary ms directory
    shutil.rmtree(tmp_ms_path)
    
else:
    print("Already transformed to ms format")

