/************************************************************************************
Copyright : Copyright (c) Facebook Technologies, LLC and its affiliates. All rights reserved.

Your use of this SDK or tool is subject to the Oculus SDK License Agreement, available at
https://developer.oculus.com/licenses/oculussdk/

Unless required by applicable law or agreed to in writing, the Utilities SDK distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
ANY KIND, either express or implied. See the License for the specific language governing
permissions and limitations under the License.
************************************************************************************/

using UnityEngine;
using UnityEngine.Events;

namespace Oculus.Interaction.Samples
{
    public class RespawnOnDrop : MonoBehaviour
    {
        [SerializeField]
        private float _yThresholdForRespawn;

        [SerializeField]
        private UnityEvent _whenRespawned = new UnityEvent();

        public UnityEvent WhenRespawned => _whenRespawned;

        // cached starting transform
        private Vector3 _initialPosition;
        private Quaternion _initialRotation;
        private Vector3 _initialScale;

        private TwoGrabFreeTransformer[] _freeTransformers;
        private Rigidbody _rigidBody;

        protected virtual void OnEnable()
        {
            _initialPosition = transform.position;
            _initialRotation = transform.rotation;
            _initialScale = transform.localScale;
            _freeTransformers = GetComponents<TwoGrabFreeTransformer>();
            _rigidBody = GetComponent<Rigidbody>();
        }

        protected virtual void Update()
        {
            if (transform.position.y < _yThresholdForRespawn)
            {
                transform.position = _initialPosition;
                transform.rotation = _initialRotation;
                transform.localScale = _initialScale;

                if (_rigidBody)
                {
                    _rigidBody.velocity = Vector3.zero;
                    _rigidBody.angularVelocity = Vector3.zero;
                }

                foreach (var freeTransformer in _freeTransformers)
                {
                    freeTransformer.MarkAsBaseScale();
                }

                _whenRespawned.Invoke();
            }
        }
    }
}
