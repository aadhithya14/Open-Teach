/************************************************************************************
Copyright : Copyright (c) Facebook Technologies, LLC and its affiliates. All rights reserved.

Your use of this SDK or tool is subject to the Oculus SDK License Agreement, available at
https://developer.oculus.com/licenses/oculussdk/

Unless required by applicable law or agreed to in writing, the Utilities SDK distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
ANY KIND, either express or implied. See the License for the specific language governing
permissions and limitations under the License.
************************************************************************************/

using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Assertions;

namespace Oculus.Interaction.HandGrab
{
    public class AutoMoveTowardsTargetProvider : MonoBehaviour, IMovementProvider
    {
        [SerializeField]
        private PoseTravelData _travellingData = PoseTravelData.DEFAULT;
        public PoseTravelData TravellingData
        {
            get
            {
                return _travellingData;
            }
            set
            {
                _travellingData = value;
            }
        }

        [SerializeField, Interface(typeof(IPointableElement))]
        private MonoBehaviour _pointableElement;
        public IPointableElement PointableElement { get; private set; }

        private bool _started;

        public List<AutoMoveTowardsTarget> _movers = new List<AutoMoveTowardsTarget>();

        protected virtual void Awake()
        {
            PointableElement = _pointableElement as IPointableElement;
        }

        protected virtual void Start()
        {
            this.BeginStart(ref _started);
            Assert.IsNotNull(_pointableElement);
            this.EndStart(ref _started);
        }

        private void LateUpdate()
        {
            for (int i = _movers.Count - 1; i >= 0; i--)
            {
                AutoMoveTowardsTarget mover = _movers[i];
                if (mover.Aborting)
                {
                    mover.Tick();
                    if (mover.Stopped)
                    {
                        _movers.Remove(mover);
                    }
                }
            }
        }

        public IMovement CreateMovement()
        {
            AutoMoveTowardsTarget mover = new AutoMoveTowardsTarget(_travellingData, PointableElement);
            mover.WhenAborted += HandleAborted;
            return mover;
        }

        private void HandleAborted(AutoMoveTowardsTarget mover)
        {
            mover.WhenAborted -= HandleAborted;
            _movers.Add(mover);
        }

        #region Inject

        public void InjectAllAutoMoveTowardsTargetProvider(IPointableElement pointableElement)
        {
            InjectPointableElement(pointableElement);
        }

        public void InjectPointableElement(IPointableElement pointableElement)
        {
            PointableElement = pointableElement;
            _pointableElement = pointableElement as MonoBehaviour;
        }
        #endregion
    }

    /// <summary>
    /// This IMovement stores the initial Pose, and in case
    /// of an aborted movement it will finish it itself.
    /// </summary>
    public class AutoMoveTowardsTarget : IMovement
    {
        private PoseTravelData _travellingData;
        private IPointableElement _pointableElement;

        public Pose Pose => _tween.Pose;
        public bool Stopped => _tween == null || _tween.Stopped;
        public bool Aborting { get; private set; }

        public Action<AutoMoveTowardsTarget> WhenAborted = delegate { };

        private UniqueIdentifier _identifier;
        public int Identifier => _identifier.ID;

        private Tween _tween;
        private Pose _target;
        private Pose _source;
        private bool _eventRegistered;

        public AutoMoveTowardsTarget(PoseTravelData travellingData, IPointableElement pointableElement)
        {
            _identifier = UniqueIdentifier.Generate();
            _travellingData = travellingData;
            _pointableElement = pointableElement;
        }

        public void MoveTo(Pose target)
        {
            AbortSelfAligment();
            _target = target;
            _tween = _travellingData.CreateTween(_source, target);
            if (!_eventRegistered)
            {
                _pointableElement.WhenPointerEventRaised += HandlePointerEventRaised;
                _eventRegistered = true;
            }
        }

        public void UpdateTarget(Pose target)
        {
            _target = target;
            _tween.UpdateTarget(_target);
        }

        public void StopAndSetPose(Pose pose)
        {
            if (_eventRegistered)
            {
                _pointableElement.WhenPointerEventRaised -= HandlePointerEventRaised;
                _eventRegistered = false;
            }

            _source = pose;
            if (_tween != null && !_tween.Stopped)
            {
                GeneratePointerEvent(PointerEvent.Hover);
                GeneratePointerEvent(PointerEvent.Select);
                Aborting = true;
                WhenAborted.Invoke(this);
            }
        }

        public void Tick()
        {
            _tween.Tick();
            if (Aborting)
            {
                GeneratePointerEvent(PointerEvent.Move);
                if (_tween.Stopped)
                {
                    AbortSelfAligment();
                }
            }
        }

        private void HandlePointerEventRaised(PointerArgs args)
        {
            if (args.PointerEvent == PointerEvent.Select || args.PointerEvent == PointerEvent.Unselect)
            {
                AbortSelfAligment();
            }
        }

        private void AbortSelfAligment()
        {
            if (Aborting)
            {
                Aborting = false;

                GeneratePointerEvent(PointerEvent.Unselect);
                GeneratePointerEvent(PointerEvent.Unhover);
            }
        }

        private void GeneratePointerEvent(PointerEvent pointerEvent)
        {
            PointerArgs args = new PointerArgs(Identifier, pointerEvent, Pose);
            _pointableElement.ProcessPointerEvent(args);
        }
    }
}
