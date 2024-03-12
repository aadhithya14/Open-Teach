/************************************************************************************
Copyright : Copyright (c) Facebook Technologies, LLC and its affiliates. All rights reserved.

Your use of this SDK or tool is subject to the Oculus SDK License Agreement, available at
https://developer.oculus.com/licenses/oculussdk/

Unless required by applicable law or agreed to in writing, the Utilities SDK distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
ANY KIND, either express or implied. See the License for the specific language governing
permissions and limitations under the License.
************************************************************************************/

using Oculus.Interaction.Input;
using UnityEngine;
using UnityEngine.Assertions;
using UnityEngine.Serialization;

namespace Oculus.Interaction
{
    public class ControllerOffset : MonoBehaviour
    {
        [SerializeField, Interface(typeof(IController))]
        private MonoBehaviour _controller;
        public IController Controller { get; private set; }

        [SerializeField]
        private Vector3 _offset;

        [SerializeField]
        private Quaternion _rotation = Quaternion.identity;

        private Pose _cachedPose = Pose.identity;

        protected bool _started = false;

        protected virtual void Awake()
        {
            Controller = _controller as IController;
        }

        protected virtual void Start()
        {
            this.BeginStart(ref _started);
            Assert.IsNotNull(Controller);
            this.EndStart(ref _started);
        }

        protected virtual void OnEnable()
        {
            if (_started)
            {
                Controller.ControllerUpdated += HandleControllerUpdated;
            }
        }

        protected virtual void OnDisable()
        {
            if (_started)
            {
                Controller.ControllerUpdated -= HandleControllerUpdated;
            }
        }

        private void HandleControllerUpdated()
        {
            if (Controller.TryGetPose(out Pose rootPose))
            {
                GetOffset(ref _cachedPose);
                _cachedPose.Postmultiply(rootPose);
                transform.SetPose(_cachedPose);
            }
        }

        public void GetOffset(ref Pose pose)
        {
            pose.position = _offset;
            pose.rotation = _rotation;
        }

        public void GetWorldPose(ref Pose pose)
        {
            pose.position = this.transform.position;
            pose.rotation = this.transform.rotation;
        }

        #region Inject

        public void InjectController(IController controller)
        {
            _controller = controller as MonoBehaviour;
            Controller = controller;
        }

        public void InjectOffset(Vector3 offset)
        {
            _offset = offset;
        }
        public void InjectRotation(Quaternion rotation)
        {
            _rotation = rotation;
        }

        public void InjectAllControllerOffset(IController controller,
            Vector3 offset, Quaternion rotation)
        {
            InjectController(controller);
            InjectOffset(offset);
            InjectRotation(rotation);
        }
        #endregion
    }
}
