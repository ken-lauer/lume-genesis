from contextlib import contextmanager
import copy
import sys
from typing import Sequence

import h5py
import matplotlib.animation
import matplotlib.pyplot as plt
import numpy as np
import prettytable
from pydantic import BaseModel
import pytest

from ... import tools
from ...tools import DisplayOptions
from ...version4 import Genesis4, Genesis4Input
from ...version4.input import (
    AlterSetup,
    Beam,
    Chicane,
    Corrector,
    Drift,
    Efield,
    Field,
    ImportBeam,
    ImportDistribution,
    ImportField,
    ImportTransformation,
    InitialParticles,
    Lattice,
    Line,
    MainInput,
    Marker,
    PhaseShifter,
    ProfileArray,
    ProfileConst,
    ProfileFile,
    ProfileGauss,
    ProfilePolynom,
    ProfileStep,
    Quadrupole,
    SequenceConst,
    SequencePolynom,
    SequencePower,
    SequenceRandom,
    Setup,
    Sponrad,
    Time,
    Track,
    Undulator,
    Wake,
    Write,
)
from ...version4.types import Reference
from ..conftest import test_artifacts, test_root

if sys.version_info >= (3, 12):
    from typing import Literal
else:
    # Pydantic specifically requires this for Python < 3.12
    from typing_extensions import Literal


run_basic = test_root / "genesis4" / "run_basic"


def create_run_basic_lattice() -> Lattice:
    elements = {
        "QHXH17": Quadrupole(L=0.101, k1=1.78),
        "QHXH18": Quadrupole(L=0.101, k1=-1.78),
        "QHXH19": Quadrupole(L=0.101, k1=1.78),
        "QHXH20": Quadrupole(L=0.101, k1=-1.78),
        "QHXH21": Quadrupole(L=0.101, k1=1.78),
        "QHXH22": Quadrupole(L=0.101, k1=-1.78),
        "QHXH23": Quadrupole(L=0.101, k1=1.78),
        "QHXH24": Quadrupole(L=0.101, k1=-1.78),
        "QHXH25": Quadrupole(L=0.101, k1=1.78),
        "QHXH26": Quadrupole(L=0.101, k1=-1.78),
        "QHXH27": Quadrupole(L=0.101, k1=1.78),
        "QHXH28": Quadrupole(L=0.101, k1=-1.78),
        "QHXH29": Quadrupole(L=0.101, k1=1.78),
        "QHXH30": Quadrupole(L=0.101, k1=-1.78),
        "QHXH31": Quadrupole(L=0.101, k1=1.78),
        "QHXH32": Quadrupole(L=0.101, k1=-1.78),
        "QHXH33": Quadrupole(L=0.101, k1=1.78),
        "QHXH34": Quadrupole(L=0.101, k1=-1.78),
        "QHXH35": Quadrupole(L=0.101, k1=1.78),
        "QHXH36": Quadrupole(L=0.101, k1=-1.78),
        "QHXH37": Quadrupole(L=0.101, k1=1.78),
        "QHXH38": Quadrupole(L=0.101, k1=-1.78),
        "QHXH39": Quadrupole(L=0.101, k1=1.78),
        "QHXH40": Quadrupole(L=0.101, k1=-1.78),
        "QHXH41": Quadrupole(L=0.101, k1=1.78),
        "QHXH42": Quadrupole(L=0.101, k1=-1.78),
        "QHXH43": Quadrupole(L=0.101, k1=1.78),
        "QHXH44": Quadrupole(L=0.101, k1=-1.78),
        "QHXH45": Quadrupole(L=0.101, k1=1.78),
        "QHXH46": Quadrupole(L=0.101, k1=-1.78),
        "QHXH47": Quadrupole(L=0.101, k1=1.78),
        "QHXH48": Quadrupole(L=0.101, k1=-1.78),
        "QHXH49": Quadrupole(L=0.101, k1=1.78),
        "QHXH50": Quadrupole(L=0.101, k1=-1.78),
        "UMAHXH17": Undulator(aw=1.7017, lambdau=0.026, nwig=130),
        "UMAHXH18": Undulator(aw=1.7017, lambdau=0.026, nwig=130),
        "UMAHXH19": Undulator(aw=1.7017, lambdau=0.026, nwig=130),
        "UMAHXH20": Undulator(aw=1.7017, lambdau=0.026, nwig=130),
        "UMAHXH21": Undulator(aw=1.7017, lambdau=0.026, nwig=130),
        "UMAHXH22": Undulator(aw=1.7017, lambdau=0.026, nwig=130),
        "UMAHXH23": Undulator(aw=1.7017, lambdau=0.026, nwig=130),
        "UMAHXH25": Undulator(aw=1.7017, lambdau=0.026, nwig=130),
        "UMAHXH26": Undulator(aw=1.7017, lambdau=0.026, nwig=130),
        "UMAHXH27": Undulator(aw=1.7017, lambdau=0.026, nwig=130),
        "UMAHXH28": Undulator(aw=1.7017, lambdau=0.026, nwig=130),
        "UMAHXH29": Undulator(aw=1.7017, lambdau=0.026, nwig=130),
        "UMAHXH30": Undulator(aw=1.7017, lambdau=0.026, nwig=130),
        "UMAHXH31": Undulator(aw=1.7017, lambdau=0.026, nwig=130),
        "UMAHXH33": Undulator(aw=1.7017, lambdau=0.026, nwig=130),
        "UMAHXH34": Undulator(aw=1.7017, lambdau=0.026, nwig=130),
        "UMAHXH35": Undulator(aw=1.7017, lambdau=0.026, nwig=130),
        "UMAHXH36": Undulator(aw=1.7017, lambdau=0.026, nwig=130),
        "UMAHXH37": Undulator(aw=1.7017, lambdau=0.026, nwig=130),
        "UMAHXH38": Undulator(aw=1.7017, lambdau=0.026, nwig=130),
        "UMAHXH39": Undulator(aw=1.7017, lambdau=0.026, nwig=130),
        "UMAHXH40": Undulator(aw=1.7017, lambdau=0.026, nwig=130),
        "UMAHXH41": Undulator(aw=1.7017, lambdau=0.026, nwig=130),
        "UMAHXH42": Undulator(aw=1.7017, lambdau=0.026, nwig=130),
        "UMAHXH43": Undulator(aw=1.7017, lambdau=0.026, nwig=130),
        "UMAHXH44": Undulator(aw=1.7017, lambdau=0.026, nwig=130),
        "UMAHXH45": Undulator(aw=1.7017, lambdau=0.026, nwig=130),
        "UMAHXH46": Undulator(aw=1.7017, lambdau=0.026, nwig=130),
        "UMAHXH47": Undulator(aw=1.7017, lambdau=0.026, nwig=130),
        "UMAHXH48": Undulator(aw=1.7017, lambdau=0.026, nwig=130),
        "UMAHXH49": Undulator(aw=1.7017, lambdau=0.026, nwig=130),
        "UMAHXH50": Undulator(aw=1.7017, lambdau=0.026, nwig=130),
        "CORR00": Corrector(L=0.001),
        "CORR01": Corrector(L=0.001),
        "CORR02": Corrector(L=0.001),
        "CORR03": Corrector(L=0.001),
        "CORR04": Corrector(L=0.001),
        "CORR05": Corrector(L=0.001),
        "CORR06": Corrector(L=0.001),
        "CORR07": Corrector(L=0.001),
        "CORR08": Corrector(L=0.001),
        "CORR09": Corrector(L=0.001),
        "CORR10": Corrector(L=0.001),
        "CORR11": Corrector(L=0.001),
        "CORR12": Corrector(L=0.001),
        "CORR13": Corrector(L=0.001),
        "CORR14": Corrector(L=0.001),
        "CORR15": Corrector(L=0.001),
        "CORR16": Corrector(L=0.001),
        "CORR17": Corrector(L=0.001),
        "CORR18": Corrector(L=0.001),
        "CORR19": Corrector(L=0.001),
        "CORR20": Corrector(L=0.001),
        "CORR21": Corrector(L=0.001),
        "CORR22": Corrector(L=0.001),
        "CORR23": Corrector(L=0.001),
        "CORR24": Corrector(L=0.001),
        "CORR25": Corrector(L=0.001),
        "CORR26": Corrector(L=0.001),
        "CORR27": Corrector(L=0.001),
        "CORR28": Corrector(L=0.001),
        "CORR29": Corrector(L=0.001),
        "CORR30": Corrector(L=0.001),
        "CORR31": Corrector(L=0.001),
        "CORR32": Corrector(L=0.001),
        "CORR33": Corrector(L=0.001),
        "D1": Drift(L=0.1335),
        "D2": Drift(L=0.4615),
        "D3": Drift(L=0.328),
        "D4": Drift(L=3.4),
        "HXRSS": Chicane(L=4.55, lb=0.1, ld=0.0, delay=0.0),
    }

    # fmt: off
    elements["HXR"] = Line.from_labels(
        elements,
        "D1", "UMAHXH17", "D2", "QHXH17", "CORR00", "D3",
        "D1", "UMAHXH18", "D2", "QHXH18", "CORR01", "D3",
        "D1", "UMAHXH19", "D2", "QHXH19", "CORR02", "D3",
        "D1", "UMAHXH20", "D2", "QHXH20", "CORR03", "D3",
        "D1", "UMAHXH21", "D2", "QHXH21", "CORR04", "D3",
        "D1", "UMAHXH22", "D2", "QHXH22", "CORR05", "D3",
        "D1", "UMAHXH23", "D2", "QHXH23", "CORR06", "D3",
        "D1", "HXRSS", "D2", "QHXH24", "CORR07", "D3",
        "D1", "UMAHXH25", "D2", "QHXH25", "CORR08", "D3",
        "D1", "UMAHXH26", "D2", "QHXH26", "CORR09", "D3",
        "D1", "UMAHXH27", "D2", "QHXH27", "CORR10", "D3",
        "D1", "UMAHXH28", "D2", "QHXH28", "CORR11", "D3",
        "D1", "UMAHXH29", "D2", "QHXH29", "CORR12", "D3",
        "D1", "UMAHXH30", "D2", "QHXH30", "CORR13", "D3",
        "D1", "UMAHXH31", "D2", "QHXH31", "CORR14", "D3",
        "D1", "D4", "D2", "QHXH32", "CORR15", "D3",
        "D1", "UMAHXH33", "D2", "QHXH33", "CORR16", "D3",
        "D1", "UMAHXH34", "D2", "QHXH34", "CORR17", "D3",
        "D1", "UMAHXH35", "D2", "QHXH35", "CORR18", "D3",
        "D1", "UMAHXH36", "D2", "QHXH36", "CORR19", "D3",
        "D1", "UMAHXH37", "D2", "QHXH37", "CORR20", "D3",
        "D1", "UMAHXH38", "D2", "QHXH38", "CORR21", "D3",
        "D1", "UMAHXH39", "D2", "QHXH39", "CORR22", "D3",
        "D1", "UMAHXH40", "D2", "QHXH40", "CORR23", "D3",
        "D1", "UMAHXH41", "D2", "QHXH41", "CORR24", "D3",
        "D1", "UMAHXH42", "D2", "QHXH42", "CORR25", "D3",
        "D1", "UMAHXH43", "D2", "QHXH43", "CORR26", "D3",
        "D1", "UMAHXH44", "D2", "QHXH44", "CORR27", "D3",
        "D1", "UMAHXH45", "D2", "QHXH45", "CORR28", "D3",
        "D1", "UMAHXH46", "D2", "QHXH46", "CORR29", "D3",
        "D1", "UMAHXH47", "D2", "QHXH47", "CORR30", "D3",
        "D1", "UMAHXH48", "D2", "QHXH48", "CORR31", "D3",
        "D1", "UMAHXH49", "D2", "QHXH49", "CORR32", "D3",
        "D1", "UMAHXH50", "D2", "QHXH50", "CORR33", "D3",
    )
    # fmt: on

    # We can also consider allowing the following to avoid quotes:
    elements["LCLS2_HXR_U1"] = Line.from_labels(
        elements,
        "D1 UMAHXH17 D2 QHXH17 CORR00 D3",
        "D1 UMAHXH18 D2 QHXH18 CORR01 D3",
        "D1 UMAHXH19 D2 QHXH19 CORR02 D3",
        "D1 UMAHXH20 D2 QHXH20 CORR03 D3",
        "D1 UMAHXH21 D2 QHXH21 CORR04 D3",
        "D1 UMAHXH22 D2 QHXH22 CORR05 D3",
        "D1 UMAHXH23 D2 QHXH23 CORR06 D3",
    )

    elements["LCLS2_HXR_U2"] = Line.from_labels(
        elements,
        "D1 UMAHXH25 D2 QHXH25 CORR08 D3",
        "D1 UMAHXH26 D2 QHXH26 CORR09 D3",
        "D1 UMAHXH27 D2 QHXH27 CORR10 D3",
        "D1 UMAHXH28 D2 QHXH28 CORR11 D3",
        "D1 UMAHXH29 D2 QHXH29 CORR12 D3",
        "D1 UMAHXH30 D2 QHXH30 CORR13 D3",
        "D1 UMAHXH31 D2 QHXH31 CORR14 D3",
        "D1       D4 D2 QHXH32 CORR15 D3",
        "D1 UMAHXH33 D2 QHXH33 CORR16 D3",
        "D1 UMAHXH34 D2 QHXH34 CORR17 D3",
        "D1 UMAHXH35 D2 QHXH35 CORR18 D3",
        "D1 UMAHXH36 D2 QHXH36 CORR19 D3",
        "D1 UMAHXH37 D2 QHXH37 CORR20 D3",
        "D1 UMAHXH38 D2 QHXH38 CORR21 D3",
        "D1 UMAHXH39 D2 QHXH39 CORR22 D3",
        "D1 UMAHXH40 D2 QHXH40 CORR23 D3",
        "D1 UMAHXH41 D2 QHXH41 CORR24 D3",
        "D1 UMAHXH42 D2 QHXH42 CORR25 D3",
        "D1 UMAHXH43 D2 QHXH43 CORR26 D3",
        "D1 UMAHXH44 D2 QHXH44 CORR27 D3",
        "D1 UMAHXH45 D2 QHXH45 CORR28 D3",
        "D1 UMAHXH46 D2 QHXH46 CORR29 D3",
        "D1 UMAHXH47 D2 QHXH47 CORR30 D3",
        "D1 UMAHXH48 D2 QHXH48 CORR31 D3",
        "D1 UMAHXH49 D2 QHXH49 CORR32 D3",
        "D1 UMAHXH50 D2 QHXH50 CORR33 D3",
    )
    return Lattice(elements=elements)


def create_run_basic_main_input() -> MainInput:
    with h5py.File(run_basic / "beam_current.h5") as fp:
        current_x = np.asarray(fp["s"])
        current = np.asarray(fp["current"])

    with h5py.File(run_basic / "beam_gamma.h5") as fp:
        gamma_x = np.asarray(fp["s"])
        gamma = np.asarray(fp["gamma"])

    main = MainInput(
        namelists=[
            Setup(
                rootname="LCLS2_HXR_9keV",
                outputdir="",
                # lattice="hxr.lat",
                beamline="HXR",
                gamma0=19174.0776,
                lambda0=1.3789244869952112e-10,
                delz=0.026,
                seed=84672,
                npart=1024,
            ),
            Time(slen=0.0000150, sample=200),
            Field(
                dgrid=0.0001,
                ngrid=101,
                harm=1,
                accumulate=True,
            ),
            # Instead of a file, we specify an array:
            # ProfileFile(
            #     label="beamcurrent",
            #     xdata="beam_current.h5/s",
            #     ydata="beam_current.h5/current",
            #     isTime=False,
            #     reverse=False,
            #     autoassign=False,
            # ),
            ProfileArray(
                label="beamcurrent",
                xdata=current_x,
                ydata=current,
            ),
            # Instead of a file, we specify an array:
            # ProfileFile(
            #     label="beamgamma",
            #     xdata="beam_gamma.h5/s",
            #     ydata="beam_gamma.h5/gamma",
            #     isTime=False,
            #     reverse=False,
            #     autoassign=False,
            # ),
            ProfileArray(
                label="beamgamma",
                xdata=gamma_x,
                ydata=gamma,
            ),
            Beam(
                # TODO: allow usage of beam_current instance directly
                current=Reference("beamcurrent"),
                gamma=Reference("beamgamma"),
                delgam=3.97848,
                ex=4.000000e-7,
                ey=4.000000e-7,
                betax=7.910909406464387,
                betay=16.8811786213468987,
                alphax=-0.7393217413918415,
                alphay=1.3870723536888105,
            ),
            Track(
                zstop=10.0,
                field_dump_at_undexit=False,
            ),
        ],
    )

    # Restrict the simulation to a small Z range for the test suite:
    main.by_namelist[Track][0].zstop = 0.1
    return main


@pytest.fixture
def lattice() -> Lattice:
    return create_run_basic_lattice()


@pytest.fixture
def main_input() -> MainInput:
    return create_run_basic_main_input()


@pytest.fixture(
    params=[
        Genesis4Input(
            main=create_run_basic_main_input(),
            lattice=create_run_basic_lattice(),
        )
    ]
)
def genesis4_input(request: pytest.FixtureRequest) -> Genesis4Input:
    return copy.deepcopy(request.param)


@pytest.fixture
def genesis4(genesis4_input: Genesis4Input) -> Genesis4:
    return Genesis4(genesis4_input)


@pytest.fixture(autouse=True, scope="function")
def _plot_show_to_savefig(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = 0

    def savefig():
        nonlocal index
        filename = test_artifacts / f"{request.node.name}_{index}.png"
        print(f"Saving figure (_plot_show_to_savefig fixture) to {filename}")
        plt.savefig(filename)
        index += 1

    monkeypatch.setattr(plt, "show", savefig)

    def plot_and_savefig(*args, **kwargs):
        res = orig_plot(*args, **kwargs)
        savefig()
        return res

    orig_plot = Genesis4.plot
    monkeypatch.setattr(Genesis4, "plot", plot_and_savefig)

    def anim_save(self, filename, *args, **kwargs):
        new_filename = test_artifacts / f"{request.node.name}_{filename}"
        return orig_anim_save(self, new_filename, *args, **kwargs)

    anim = matplotlib.animation.FuncAnimation
    orig_anim_save = anim.save
    monkeypatch.setattr(anim, "save", anim_save)


def get_elements() -> Sequence[BaseModel]:
    return [
        Chicane(),
        Corrector(),
        Drift(),
        Marker(),
        PhaseShifter(),
        Quadrupole(),
        Undulator(),
        AlterSetup(),
        Beam(),
        Efield(),
        Field(),
        ImportBeam(),
        ImportDistribution(),
        ImportField(),
        ImportTransformation(),
        ProfileConst(label="label"),
        ProfileFile(label="label"),
        ProfileGauss(label="label"),
        ProfilePolynom(label="label"),
        ProfileStep(label="label"),
        SequenceConst(label="label"),
        SequencePolynom(label="label"),
        SequencePower(label="label"),
        SequenceRandom(label="label"),
        Setup(),
        Sponrad(),
        Time(),
        Track(),
        Wake(),
        Write(),
        # InitialParticles(filename='...'),
        Lattice(),
        Line(),
        ProfileArray(label="label", xdata=[0.0], ydata=[0.0]),
        InitialParticles(
            data={
                "x": np.asarray([0.0]),
                "y": np.asarray([0.0]),
                "z": np.asarray([0.0]),
                "px": np.asarray([0.0]),
                "py": np.asarray([0.0]),
                "pz": np.asarray([0.0]),
                "t": np.asarray([0.0]),
                "status": np.asarray([0.0]),
                "weight": np.asarray([0.0]),
                "species": "species",
            }
        ),
    ]


@pytest.fixture(
    params=[elem for elem in get_elements()],
    ids=[type(elem).__name__ for elem in get_elements()],
)
def element(request: pytest.FixtureRequest):
    return request.param


@contextmanager
def display_options_ctx(
    jupyter_render_mode: Literal["html", "markdown", "genesis"] = "html",
    console_render_mode: Literal["markdown", "genesis"] = "markdown",
    echo_genesis_output: bool = True,
    include_description: bool = True,
    filter_tab_completion: bool = True,
    ascii_table_type: int = prettytable.MARKDOWN,
):
    tools.global_display_options = DisplayOptions(
        jupyter_render_mode=jupyter_render_mode,
        console_render_mode=console_render_mode,
        echo_genesis_output=echo_genesis_output,
        include_description=include_description,
        filter_tab_completion=filter_tab_completion,
        ascii_table_type=ascii_table_type,
    )
    yield
    tools.global_display_options = DisplayOptions()
